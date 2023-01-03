{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
module Main (main) where

import Control.Applicative ((<|>), many)
import Control.Monad (foldM_, forM_)
import Data.Attoparsec.Text (Parser)
import Data.ByteString.Lazy (ByteString)
import Data.Digest.Pure.MD5 (MD5Digest, md5)
import Data.Foldable (foldlM)
import Data.Map (Map)
import Data.Text (Text)
import Data.Text.Encoding (encodeUtf8)
import Data.Typeable (Typeable)
import Data.YAML ((.=), ToYAML(..))
import Foreign.C.String (CString, peekCString, withCString)
import Numeric (showHex)
import Path ((</>), Abs, Dir, File, Path, Rel, parent, toFilePath)
import Path.IO (createFileLink, doesFileExist, ensureDir, listDir, resolveDir', resolveFile')
import Replace.Attoparsec.Text (streamEdit)
import System.FilePath (takeBaseName)
import Text.Printf (printf)
import Text.Regex (mkRegex, subRegex)
import Text.Regex.TDFA ((=~))
import Text.Slugify (slugifyUnicode)

import qualified Data.Aeson.Micro as JSON
import qualified Data.Attoparsec.Text as A
import qualified Data.ByteString.Lazy as LB
import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.Maybe as MB
import qualified Data.Text as T
import qualified Data.YAML as Y
-- import qualified Data.ProtocolBuffers as PB
import qualified Database.SQLite.Simple as SQL
import qualified Lib.Git as Git
import qualified Network.URI.Encode as URI
import qualified Path.Internal

foreign import ccall "htidy.h tidy" tidy' :: CString -> IO CString

tidy :: String -> IO String
tidy s = withCString s tidy' >>= peekCString

maxFilenameSize :: Int
maxFilenameSize = 60

b91Alphas :: Text
b91Alphas = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

b91Symbols :: Text
b91Symbols = "!#$%&()*+,-./:;<=>?@[]^_`{|}~"

b91s :: Text
b91s = b91Alphas <> b91Symbols

-- ==================== Types, Typeclasses, and Instances ====================


-- Extra base types for `Path` to distinguish between paths that exist (i.e.
-- there is a file or directory there) and paths that do not.
data Extant
  deriving Typeable

-- Nid, Mid, Guid, Tags, Flds, SortFld
data SQLNote = SQLNote !Integer !Integer !Text !Text !Text !Text
  deriving Show

-- Cid, Nid, Did, Ord
data SQLCard = SQLCard !Integer !Integer !Integer !Integer
  deriving Show

-- Did, Name
data SQLDeck = SQLDeck !Integer !Text
  deriving (Eq, Ord, Show)

data SQLModel = SQLModel
  { sqlModelMid  :: !Integer
  , sqlModelName :: !Text
  , sqlModelConfigHex :: !Text
  }
  deriving Show

data SQLField = SQLField
  { sqlFieldMid  :: !Integer
  , sqlFieldOrd  :: !Integer
  , sqlFieldName :: !Text
  , sqlFieldConfigHex :: !Text
  }
  deriving Show

data SQLTemplate = SQLTemplate
  { sqlTemplateMid  :: !Integer
  , sqlTemplateOrd  :: !Integer
  , sqlTemplateName :: !Text
  , sqlTemplateConfigHex :: !Text
  }
  deriving Show

newtype Mid = Mid Integer deriving (Eq, Ord, Show, ToYAML)

newtype Cid = Cid Integer
newtype Did = Did Integer deriving (Eq, Ord)
newtype Nid = Nid Integer deriving (Eq, Ord)
newtype Guid = Guid Text
newtype Tags = Tags [Text]
newtype Fields = Fields [Field]
newtype DeckName = DeckName Text
newtype SortField = SortField Text
newtype ModelName = ModelName Text deriving (ToYAML)

data Field = Field !FieldOrd !FieldName !Text

newtype Ord' = Ord' Integer

type Filename = Text
data MdNote = MdNote !Guid !ModelName !Tags !Fields !SortField
data ColNote = ColNote !MdNote !Nid !Filename
data Card = Card !Cid !Nid !Did !Ord' !DeckName !ColNote
data Deck = Deck ![Text] !Did
  deriving Eq

instance Ord Deck where
  (Deck parts did) <= (Deck parts' did') = parts <= parts' && did <= did'

instance SQL.FromRow SQLNote where
  fromRow =
    SQLNote <$> SQL.field <*> SQL.field <*> SQL.field <*> SQL.field <*> SQL.field <*> SQL.field

instance SQL.FromRow SQLCard where
  fromRow = SQLCard <$> SQL.field <*> SQL.field <*> SQL.field <*> SQL.field

instance SQL.FromRow SQLDeck where
  fromRow = SQLDeck <$> SQL.field <*> SQL.field

instance SQL.FromRow SQLModel where
  fromRow = SQLModel <$> SQL.field <*> SQL.field <*> SQL.field

instance SQL.FromRow SQLField where
  fromRow = SQLField <$> SQL.field <*> SQL.field <*> SQL.field <*> SQL.field

instance SQL.FromRow SQLTemplate where
  fromRow = SQLTemplate <$> SQL.field <*> SQL.field <*> SQL.field <*> SQL.field

newtype FieldOrd = FieldOrd Integer deriving (Eq, Ord, ToYAML)
newtype FieldName = FieldName Text deriving (Eq, Ord, ToYAML)
newtype TemplateName = TemplateName Text deriving ToYAML

data FieldDef = FieldDef !Mid !FieldOrd !FieldName
  deriving Eq
data Template = Template !Mid !FieldOrd !TemplateName
data Model = Model !Mid !ModelName !(Map FieldOrd (FieldDef, Template))

instance Ord FieldDef where
  (FieldDef _ (FieldOrd ord) _) <= (FieldDef _ (FieldOrd ord') _) = ord <= ord'

instance ToYAML FieldDef where
  toYAML (FieldDef mid ord name) = Y.mapping ["mid" .= mid, "ord" .= ord, "name" .= name]

instance ToYAML Template where
  toYAML (Template mid ord name) = Y.mapping ["mid" .= mid, "ord" .= ord, "name" .= name]

instance ToYAML Model where
  toYAML (Model mid ord fieldsAndTemplatesByOrd) =
    Y.mapping ["mid" .= mid, "ord" .= ord, "fieldsAndTemplatesByOrd" .= fieldsAndTemplatesByOrd]

data Repo = Repo !(Path Extant Dir) !Git.Config


-- ========================== Path utility functions ==========================


-- | Ensure a directory is empty, creating it if necessary, and returning
-- `Nothing` if it was not.
ensureEmpty :: Path Abs Dir -> IO (Maybe (Path Extant Dir))
ensureEmpty dir = do
  ensureDir dir
  contents <- listDir dir
  pure $ if contents == ([], []) then Just $ Path.Internal.Path (toFilePath dir) else Nothing


ensureExtantDir :: Path Abs Dir -> IO (Path Extant Dir)
ensureExtantDir dir = do
  ensureDir dir
  pure $ Path.Internal.Path (toFilePath dir)


getExtantFile :: Path Abs File -> IO (Maybe (Path Extant File))
getExtantFile file = do
  exists <- doesFileExist file
  if exists then pure $ Just $ Path.Internal.Path (toFilePath file) else pure Nothing


-- | Convert from an `Extant` or `Missing` path to an `Abs` path.
absify :: Path a b -> Path Abs b
absify (Path.Internal.Path s) = Path.Internal.Path s


mkFileLink :: Path Extant File -> Path Abs File -> IO (Path Extant File)
mkFileLink tgt lnk = do
  createFileLink tgt lnk
  pure $ ((Path.Internal.Path . T.unpack) . T.pack . toFilePath) lnk


mkNewFile :: Path Extant Dir -> Text -> Text -> IO (Path Extant File)
mkNewFile dir name contents = do
  writeFile (toFilePath file) (T.unpack contents)
  pure file
  where file = dir </> (Path.Internal.Path . T.unpack) name


-- ======================= Repository utility functions =======================


gitCommitAll :: Path Extant Dir -> String -> IO Repo
gitCommitAll root msg = do
  Git.runGit gitConfig (Git.initDB False >> Git.add ["."] >> Git.commit [] "Ki" "ki" msg [])
  pure (Repo root gitConfig)
  where gitConfig = Git.makeConfig (toFilePath root) Nothing

-- | Clone a local repository into the given directory, which must exist and be
-- empty (data invariant).
gitClone :: Repo -> Path Extant Dir -> IO Repo
gitClone (Repo root config) tgt = do
  Git.runGit config $ do
    o <- Git.gitExec "clone" [toFilePath root, toFilePath tgt] []
    case o of
      Right _   -> pure ()
      Left  err -> Git.gitError err "clone"
  pure $ Repo tgt $ Git.makeConfig (toFilePath tgt) Nothing


-- ================================ Core logic ================================


-- | Convert a list of raw `SQLDeck`s into a preorder traversal of components.
-- This can be reversed to get a postorder traversal.
mkDeckTree :: [SQLDeck] -> [Deck]
mkDeckTree = L.sort . map unpack
  where
    unpack :: SQLDeck -> Deck
    unpack (SQLDeck did fullname) = Deck (T.splitOn "::" fullname) (Did did)


mapFieldsByMid :: [FieldDef] -> Map Mid (Map FieldOrd FieldDef)
mapFieldsByMid = foldr f M.empty
  where
    g :: FieldOrd -> FieldDef -> Map FieldOrd FieldDef -> Maybe (Map FieldOrd FieldDef)
    g ord fld fieldsByOrd = Just $ M.insert ord fld fieldsByOrd
    f :: FieldDef -> Map Mid (Map FieldOrd FieldDef) -> Map Mid (Map FieldOrd FieldDef)
    f fld@(FieldDef mid ord _) = M.update (g ord fld) mid


mapTemplatesByMid :: [Template] -> Map Mid (Map FieldOrd Template)
mapTemplatesByMid = foldr f M.empty
  where
    g :: FieldOrd -> Template -> Map FieldOrd Template -> Maybe (Map FieldOrd Template)
    g ord tmpl fieldsByOrd = Just $ M.insert ord tmpl fieldsByOrd
    f :: Template -> Map Mid (Map FieldOrd Template) -> Map Mid (Map FieldOrd Template)
    f tmpl@(Template mid ord _) = M.update (g ord tmpl) mid


stripHtmlTags :: String -> String
stripHtmlTags "" = ""
stripHtmlTags ('<' : xs) = stripHtmlTags $ drop 1 $ dropWhile (/= '>') xs
stripHtmlTags (x : xs) = x : stripHtmlTags xs


subRegex' :: Text -> Text -> Text -> Text
subRegex' pat rep t = T.pack $ subRegex (mkRegex pat') t' rep'
  where
    pat' = T.unpack pat
    rep' = T.unpack rep
    t'   = T.unpack t


plainToHtml :: Text -> Text
plainToHtml s
  | t =~ htmlRegex = T.replace "\n" "<br>" t
  | otherwise = t
  where
    htmlRegex = "</?\\s*[a-z-][^>]*\\s*>|(\\&(?:[\\w\\d]+|#\\d+|#x[a-f\\d]+);)" :: Text
    sub :: Text -> Text
    sub =
      subRegex' "<div>\\s*</div>" ""
        . subRegex' "<i>\\s*</i>" ""
        . subRegex' "<b>\\s*</b>" ""
        . T.replace "&nbsp;" " "
        . T.replace "&amp;" "&"
        . T.replace "&gt;" ">"
        . T.replace "&lt;" "<"
    t = sub s


getModel :: Map Mid (Map FieldOrd (FieldDef, Template)) -> SQLModel -> Maybe Model
getModel fieldsAndTemplatesByMid (SQLModel mid name _) =
  case M.lookup (Mid mid) fieldsAndTemplatesByMid of
    Just m  -> Just (Model (Mid mid) (ModelName name) m)
    Nothing -> Nothing


getFieldsAndTemplatesByMid :: Map Mid (Map FieldOrd FieldDef)
                           -> Map Mid (Map FieldOrd Template)
                           -> Map Mid (Map FieldOrd (FieldDef, Template))
getFieldsAndTemplatesByMid fieldsByMid templatesByMid = M.foldrWithKey f M.empty fieldsByMid
  where
    f :: Mid
      -> Map FieldOrd FieldDef
      -> Map Mid (Map FieldOrd (FieldDef, Template))
      -> Map Mid (Map FieldOrd (FieldDef, Template))
    f mid fieldsByOrd acc = case M.lookup mid templatesByMid of
      Just templatesByOrd -> M.insert mid (M.intersectionWith (,) fieldsByOrd templatesByOrd) acc
      Nothing -> acc


getField :: SQLField -> FieldDef
getField (SQLField mid ord name _) = FieldDef (Mid mid) (FieldOrd ord) (FieldName name)


getTemplate :: SQLTemplate -> Template
getTemplate (SQLTemplate mid ord name _) = Template (Mid mid) (FieldOrd ord) (TemplateName name)

mkSlug :: Text -> Text
mkSlug = slugifyUnicode . T.take maxFilenameSize . T.pack . stripHtmlTags . T.unpack . plainToHtml


-- | Get the hex representation of a note's GUID.
--
-- If for some reason we get a failure of `T.findIndex`, i.e. we have some
-- character that is not a valid base91 char, then we fallback to taking the
-- md5sum of the UTF-8 encoded text of the GUID.
guidToHex :: Text -> Text
guidToHex guid = case val of
  Just x  -> T.pack $ showHex x ""
  Nothing -> T.pack $ show $ md5 $ LB.fromStrict $ encodeUtf8 guid
  where
    digits = mapM (\c -> T.findIndex (== c) b91s) (T.unpack guid)
    val    = L.foldl' (\acc x -> acc * 91 + x) 0 <$> digits


-- | Construct a filename (without extension) for a given markdown note.
--
-- We first try to construct it solely from the sort field. If that fails
-- (yields an empty string), then we try concatenating all the fields together,
-- and if that fails, we fall back to concatenating the model name and the hex
-- representation of the guid.
mkFilename :: MdNote -> Text
mkFilename (MdNote (Guid guid) (ModelName model) _ (Fields fields) (SortField sfld))
  | (not . T.null) short = short
  | (not . T.null) long = long
  | otherwise = fallback
  where
    short    = mkSlug sfld
    long     = mkSlug $ T.concat $ map (\(Field _ _ s) -> s) fields
    fallback = model <> "--" <> guidToHex guid


zipEq :: [a] -> [b] -> Maybe [(a, b)]
zipEq [] [] = Just []
zipEq [] _  = Nothing
zipEq _  [] = Nothing
zipEq (x : xs) (y : ys) = ((x, y) :) <$> zipEq xs ys


mkColNote :: Map Mid Model -> SQLNote -> Maybe ColNote
mkColNote modelsByMid (SQLNote nid mid guid tags flds sfld) = do
  (Model _ modelName modelFieldsAndTemplatesByOrd) <- M.lookup (Mid mid) modelsByMid
  fields <- map mkField <$> (zipEq fs . L.sort . M.elems . M.map fst) modelFieldsAndTemplatesByOrd
  let mdNote = MdNote (Guid guid) modelName (Tags ts) (Fields fields) (SortField sfld)
  pure (ColNote mdNote (Nid nid) (mkFilename mdNote))
  where
    ts = T.words tags
    fs = T.split (== '\x1f') flds
    mkField :: (Text, FieldDef) -> Field
    mkField (s, FieldDef _ ord name) = Field ord name s


getCard :: Map Nid ColNote -> [SQLDeck] -> SQLCard -> Maybe Card
getCard colnotesByNid decks (SQLCard cid nid did ord) =
  Card (Cid cid) (Nid nid) (Did did) (Ord' ord)
    .   DeckName
    <$> (M.lookup did . M.fromList . map unpack) decks
    <*> M.lookup (Nid nid) colnotesByNid
  where
    unpack :: SQLDeck -> (Integer, Text)
    unpack (SQLDeck did' name) = (did', name)


-- Parse the collection and target directory, then call `continueClone`.
clone :: String -> String -> IO ()
clone colPath targetPath = do
  maybeColFile   <- resolveFile' colPath >>= getExtantFile
  maybeTargetDir <- resolveDir' targetPath >>= ensureEmpty
  case (maybeColFile, maybeTargetDir) of
    (Nothing, _) -> printf "fatal: collection file '%s' does not exist" (show colPath)
    (_, Nothing) -> printf "fatal: targetdir '%s' not empty" (show targetPath)
    (Just colFile, Just targetDir) -> continueClone colFile targetDir


-- | This grabs the filename without the file extension, appends `.media`, and
-- then converts it back to a path.
--
-- We could use `takeBaseName` instead.
ankiMediaDirname :: Path Extant File -> Path Rel Dir
ankiMediaDirname colFile = Path.Internal.Path $ stem ++ ".media"
  where stem = (takeBaseName . toFilePath) colFile


continueClone :: Path Extant File -> Path Extant Dir -> IO ()
continueClone colFile targetDir = do
  -- Hash the collection file.
  colFileContents <- LB.readFile (toFilePath colFile)
  let colFileMD5 = md5 colFileContents
  -- Add the backups directory to the `.gitignore` file.
  writeFile (toFilePath gitIgnore) ".ki/backups"
  -- Create `.ki` and `_media` subdirectories.
  maybeKiDir     <- ensureEmpty (absify targetDir </> Path.Internal.Path ".ki")
  maybeMediaDir  <- ensureEmpty (absify targetDir </> Path.Internal.Path "_media")
  maybeModelsDir <- ensureEmpty (absify targetDir </> Path.Internal.Path "_models")
  ankiMediaDir   <- ensureExtantDir (absify ankiUserDir </> ankiMediaDirname colFile)
  case (maybeKiDir, maybeMediaDir, maybeModelsDir) of
    (Nothing, _, _) -> printf "fatal: new '.ki' directory not empty"
    (_, Nothing, _) -> printf "fatal: new '_media' directory not empty"
    (_, _, Nothing) -> printf "fatal: new '_models' directory not empty"
    -- Write repository contents and commit.
    (Just kiDir, Just mediaDir, Just modelsDir) ->
      writeInitialCommit colFile targetDir kiDir mediaDir ankiMediaDir modelsDir colFileMD5
  where
    gitIgnore   = absify targetDir </> Path.Internal.Path ".gitignore" :: Path Abs File
    ankiUserDir = parent colFile :: Path Extant Dir


writeInitialCommit :: Path Extant File
                   -> Path Extant Dir
                   -> Path Extant Dir
                   -> Path Extant Dir
                   -> Path Extant Dir
                   -> Path Extant Dir
                   -> MD5Digest
                   -> IO ()
writeInitialCommit colFile targetDir kiDir mediaDir ankiMediaDir modelsDir _ = do
  _ <- writeRepo colFile targetDir kiDir mediaDir ankiMediaDir modelsDir
  _ <- gitCommitAll targetDir "Initial commit."
  pure ()


writeRepo :: Path Extant File
          -> Path Extant Dir
          -> Path Extant Dir
          -> Path Extant Dir
          -> Path Extant Dir
          -> Path Extant Dir
          -> IO ()
writeRepo colFile targetDir kiDir mediaDir ankiMediaDir modelsDir = do
  LB.writeFile (toFilePath $ kiDir </> Path.Internal.Path "config") $ JSON.encode config
  conn  <- SQL.open (toFilePath colFile)
  ns    <- SQL.query_ conn "SELECT (nid,guid,mid,tags,flds,sfld) FROM notes" :: IO [SQLNote]
  cs    <- SQL.query_ conn "SELECT (id,nid,did,ord) FROM cards" :: IO [SQLCard]
  ds    <- SQL.query_ conn "SELECT (id,name) FROM decks" :: IO [SQLDeck]
  nts   <- SQL.query_ conn "SELECT (id,name,config) FROM notetypes" :: IO [SQLModel]
  flds  <- SQL.query_ conn "SELECT (ntid,ord,name,config) FROM fields" :: IO [SQLField]
  tmpls <- SQL.query_ conn "SELECT (ntid,ord,name,config) FROM templates" :: IO [SQLTemplate]
  let
    fieldsByMid = (mapFieldsByMid . map getField) flds
    templatesByMid = (mapTemplatesByMid . map getTemplate) tmpls
    fieldsAndTemplatesByMid = getFieldsAndTemplatesByMid fieldsByMid templatesByMid
    models    = map (getModel fieldsAndTemplatesByMid) nts
    modelsByMid = M.fromList (MB.mapMaybe (fmap (\m@(Model mid _ _) -> (mid, m))) models)
    colnotesByNid = M.fromList $ MB.mapMaybe (fmap unpack . mkColNote modelsByMid) ns
    serializedModelsByMid = M.map (Y.encode . (: [])) modelsByMid
    cards     = MB.mapMaybe (getCard colnotesByNid ds) cs
    preorder  = mkDeckTree ds
    postorder = reverse preorder
  ankiMediaRepo <- gitCommitAll ankiMediaDir "Initial commit."
  _kiMediaRepo  <- gitClone ankiMediaRepo mediaDir
  -- Dump all models to top-level `_models` subdirectory.
  forM_ (M.assocs serializedModelsByMid) (writeModel modelsDir)
  forM_ postorder (writeDeck targetDir cards)
  where
    remote = JSON.Object $ M.singleton "path" $ (JSON.String . T.pack . toFilePath) colFile
    config = JSON.Object $ M.singleton "remote" remote

    unpack :: ColNote -> (Nid, ColNote)
    unpack c@(ColNote _ nid _) = (nid, c)


writeModel :: Path Extant Dir -> (Mid, ByteString) -> IO ()
writeModel modelsDir (mid, s) =
  LB.writeFile (toFilePath $ modelsDir </> Path.Internal.Path (show mid ++ ".yaml")) s


writeDeck :: Path Extant Dir -> [Card] -> Deck -> IO ()
writeDeck targetDir cards deck@(Deck _ did) = do
  foldM_ (writeCard targetDir deck) M.empty deckCards
  where deckCards = filter (\(Card _ _ did' _ _ _) -> did' == did) cards

-- | Write an Anki card to disk in a target directory.
--
-- Here, we must check if we've written a file for this note before. If not,
-- then we write one and move on. If we have, then there are two remaining
-- cases:
-- - We've written a file in this deck before, in which case we should have
--    some kind of reference or handle to it.
-- - We've written a file but not in this deck, in which case we must write a
--    link to the extant file.
writeCard :: Path Extant Dir
          -> Deck
          -> Map Nid (Map Did (Path Extant File))
          -> Card
          -> IO (Map Nid (Map Did (Path Extant File)))
writeCard targetDir (Deck parts did) noteFilesByDidByNid (Card _ nid _ _ _ (ColNote mdnote _ filename))
  = do
    deckDir <- ensureExtantDir (mkDeckDir targetDir parts)
    payload <- mkPayload mdnote
    case M.lookup nid noteFilesByDidByNid of
      Nothing -> do
        file <- mkNewFile deckDir filename payload
        pure $ M.insert nid (M.singleton did file) noteFilesByDidByNid
      Just noteFilesByDid ->
        case (M.lookup did noteFilesByDid, fst <$> M.minView noteFilesByDid) of
          (Just _ , _) -> pure noteFilesByDidByNid
          (Nothing, Just noteFile) -> do
            link <- mkFileLink noteFile (mkNotePath deckDir filename)
            pure $ M.insert nid (M.insert did link noteFilesByDid) noteFilesByDidByNid
          (Nothing, Nothing) -> pure noteFilesByDidByNid


htmlToScreen :: Text -> Text
htmlToScreen =
  T.strip
    . subRegex' "\\<b\\>\\s*\\<\\/b\\>" ""
    . subRegex' "src= ?\n\"" "src=\""
    . T.replace "<br />" "\n"
    . T.replace "<br/>" "\n"
    . T.replace "<br>" "\n"
    . T.replace "&nbsp;" " "
    . T.replace "&amp;" "&"
    . T.replace "&gt;" ">"
    . T.replace "&lt;" "<"
    . T.replace "\\*}" "*}"
    . T.replace "\\\\}" "\\}"
    . T.replace "\\\\{" "\\{"
    . T.replace "\\\\\\\\" "\\\\"
    . subRegex' "\\<style\\>(?s:.)*\\<\\/style\\>" ""


escapeMediaFilenames :: Text -> Text
escapeMediaFilenames = streamEdit mediaTagParser mediaTagEditor

type Prefix = Text
type Suffix = Text
data MediaFilename = MediaFilename !Filename !Prefix !Suffix
data MediaTag = MediaTag !Filename !Prefix !Suffix

mediaTagEditor :: MediaTag -> Text
mediaTagEditor (MediaTag filename prefix suffix) = prefix <> URI.decodeText filename <> suffix

mediaTagParser :: Parser MediaTag
mediaTagParser = do
  tagName <- A.asciiCI "<" >> A.asciiCI "img" <|> A.asciiCI "audio" <|> A.asciiCI "object"
  tagSuffixHead <- T.singleton <$> A.satisfy (A.notInClass "a-zA-Z_")
  tagSuffixPith <- T.pack <$> A.many1 (A.notChar '>')
  tagSuffixLast <- T.singleton <$> A.satisfy (A.notInClass "a-zA-Z_")
  attr    <- A.asciiCI "src=" <|> A.asciiCI "data="
  (MediaFilename fname pre suf) <-
    (dubQuotedFilenameParser <|> quotedFilenameParser <|> unquotedFilenameParser) <* A.char '>'
  pure $ MediaTag
    fname
    ("<" <> tagName <> tagSuffixHead <> tagSuffixPith <> tagSuffixLast <> attr <> pre)
    (suf <> ">")


restOfTagParser :: Parser Text
restOfTagParser = do
  T.pack <$> many (A.notChar '>')


dubQuotedFilenameParser :: Parser MediaFilename
dubQuotedFilenameParser = do
  filename <- A.char '"' >> T.pack <$> A.many1 (A.notChar '"') <* A.char '"'
  rest     <- restOfTagParser
  pure $ MediaFilename filename "\"" ("\"" <> rest)


quotedFilenameParser :: Parser MediaFilename
quotedFilenameParser = do
  filename <- A.char '\'' >> T.pack <$> A.many1 (A.notChar '\'') <* A.char '\''
  rest     <- restOfTagParser
  pure $ MediaFilename filename "'" ("'" <> rest)


unquotedFilenameParser :: Parser MediaFilename
unquotedFilenameParser = do
  filename <- T.pack <$> A.many1 (A.satisfy $ A.notInClass " >")
  rest     <- A.option "" spaceAndRestParser
  pure $ MediaFilename filename "" rest
  where
    spaceAndRestParser :: Parser Text
    spaceAndRestParser = do
      rest <- A.asciiCI "\x20" >> T.pack <$> many (A.notChar '>')
      pure $ " " <> rest


mkPayload :: MdNote -> IO Text
mkPayload (MdNote (Guid guid) (ModelName modelName) (Tags tags) (Fields fields) _) = do
  body <- foldlM go "" fields
  pure $ header <> "\n" <> body
  where
    header =
      T.intercalate "\n"
        $  [ "# Note"
           , "```"
           , "guid: " <> guid
           , "notetype: " <> modelName
           , "```"
           , ""
           , "### Tags"
           , "```"
           ]
        ++ tags
        ++ ["```", ""]
    go :: Text -> Field -> IO Text
    go s (Field _ (FieldName name) text) = do
      tidyText <- T.pack <$> (tidy . T.unpack) text
      pure $ s <> "\n## " <> name <> "\n" <> (escapeMediaFilenames . htmlToScreen) tidyText <> "\n"


mkDeckDir :: Path Extant Dir -> [Text] -> Path Abs Dir
mkDeckDir targetDir [] = absify targetDir
mkDeckDir targetDir (p : ps) =
  absify targetDir
    </> foldr ((</>) . Path.Internal.Path . T.unpack) ((Path.Internal.Path . T.unpack) p) ps


mkNotePath :: Path Extant Dir -> Text -> Path Abs File
mkNotePath dir filename = absify dir </> (Path.Internal.Path . T.unpack) filename


main :: IO ()
main = do
  print ("Hello" :: String)
  clone "" ""
