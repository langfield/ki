{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
module Main (main) where

import Control.Applicative ((<|>), many)
import Control.Monad (foldM_, forM_, when)
import Data.Attoparsec.Combinator (lookAhead)
import Data.Attoparsec.Text (Parser)
import Data.ByteString.Lazy (ByteString)
import Data.Digest.Pure.MD5 (MD5Digest, md5)
import Data.Map (Map)
import Data.Text (Text)
import Data.Text.Encoding (encodeUtf8)
import Data.Typeable (Typeable)
import Data.YAML ((.=), ToYAML(..))
import Database.SQLite.Simple (SQLData(..))
import Numeric (showHex)
import Path ((</>), Abs, Dir, File, Path, Rel, parent, toFilePath)
import Path.IO (createFileLink, doesFileExist, ensureDir, listDir, resolveDir', resolveFile')
import Replace.Attoparsec.Text (streamEdit)
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.FilePath (takeBaseName)
import System.ProgressBar
  ( Progress(..)
  , ProgressBar
  , ProgressBarWidth(..)
  , Style(..)
  , defStyle
  , incProgress
  , newProgressBar
  )
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
import qualified Database.SQLite.Simple as SQL
import qualified Lib.Git as Git
import qualified Network.URI.Encode as URI
import qualified Path.Internal

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
data SQLNote = SQLNote !Integer !Integer !Text !Text !Text !SQLData
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
  , sqlModelConfigBytes :: !ByteString
  }
  deriving Show

data SQLField = SQLField
  { sqlFieldMid  :: !Integer
  , sqlFieldOrd  :: !Integer
  , sqlFieldName :: !Text
  , sqlFieldConfigBytes :: !ByteString
  }
  deriving Show

data SQLTemplate = SQLTemplate
  { sqlTemplateMid  :: !Integer
  , sqlTemplateOrd  :: !Integer
  , sqlTemplateName :: !Text
  , sqlTemplateConfigBytes :: !ByteString
  }
  deriving Show

newtype Mid = Mid Integer deriving (Eq, Ord, Show, ToYAML)

data Field = Field !FieldOrd !FieldName !Text
  deriving Show

newtype Cid = Cid Integer deriving Show
newtype Did = Did Integer deriving (Eq, Ord, Show)
newtype Nid = Nid Integer deriving (Eq, Ord, Show)
newtype Guid = Guid Text deriving Show
newtype Tags = Tags [Text] deriving Show
newtype Fields = Fields [Field] deriving Show
newtype DeckName = DeckName Text deriving Show
newtype SortField = SortField Text deriving Show
newtype ModelName = ModelName Text deriving (ToYAML)

instance Show ModelName where
  show (ModelName name) = T.unpack name

newtype Ord' = Ord' Integer deriving Show

type Filename = Text
data MdNote = MdNote !Guid !ModelName !Tags !Fields !SortField
  deriving Show
data ColNote = ColNote !MdNote !Nid !Filename
  deriving Show
data Card = Card !Cid !Nid !Did !Ord' !DeckName !ColNote
  deriving Show
data Deck = Deck ![Text] !Did
  deriving (Eq, Show)

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

newtype FieldOrd = FieldOrd Integer deriving (Eq, Ord, Show, ToYAML)
newtype FieldName = FieldName Text deriving (Eq, Ord, Show, ToYAML)
newtype TemplateOrd = TemplateOrd Integer deriving (Eq, Ord, Show, ToYAML)
newtype TemplateName = TemplateName Text deriving (Show, ToYAML)

data FieldDef = FieldDef !Mid !FieldOrd !FieldName
  deriving (Eq, Show)
data Template = Template !Mid !TemplateOrd !TemplateName
  deriving Show
data Model = Model !Mid !ModelName !(Map FieldOrd FieldDef) !(Map TemplateOrd Template)
  deriving Show

instance Ord FieldDef where
  (FieldDef _ (FieldOrd ord) _) <= (FieldDef _ (FieldOrd ord') _) = ord <= ord'

instance ToYAML FieldDef where
  toYAML (FieldDef mid ord name) = Y.mapping ["mid" .= mid, "ord" .= ord, "name" .= name]

instance ToYAML Template where
  toYAML (Template mid ord name) = Y.mapping ["mid" .= mid, "ord" .= ord, "name" .= name]

instance ToYAML Model where
  toYAML (Model mid ord fieldsByOrd templatesByOrd) = Y.mapping
    ["mid" .= mid, "ord" .= ord, "fieldsByOrd" .= fieldsByOrd, "templatesByOrd" .= templatesByOrd]

data Repo = Repo !(Path Extant Dir) !Git.Config

type Prefix = Text
type Suffix = Text
data MediaFilename = MediaFilename !Filename !Prefix !Suffix
  deriving Show
data MediaTag = MediaTag !Filename !Prefix !Suffix
  deriving Show


-- ========================== Path utility functions ==========================


mkInternalDir :: Text -> Path a Dir
mkInternalDir s = Path.Internal.Path (T.unpack withSlash)
  where
    stripped  = subRegex' "\\/+$" "" s
    withSlash = if T.null stripped then "./" else stripped <> "/"


castDir :: Path b Dir -> Path b' Dir
castDir = mkInternalDir . T.pack . toFilePath


-- | Ensure a directory is empty, creating it if necessary, and returning
-- `Nothing` if it was not.
ensureEmpty :: Path Abs Dir -> IO (Maybe (Path Extant Dir))
ensureEmpty dir = do
  ensureDir dir
  contents <- listDir dir
  pure $ if contents == ([], []) then Just $ castDir dir else Nothing


ensureExtantDir :: Path Abs Dir -> IO (Path Extant Dir)
ensureExtantDir dir = do
  ensureDir dir
  pure $ castDir dir


getExtantFile :: Path Abs File -> IO (Maybe (Path Extant File))
getExtantFile file = do
  exists <- doesFileExist file
  if exists then pure $ Just $ Path.Internal.Path (toFilePath file) else pure Nothing


-- | Convert from an `Extant` or `Missing` directory to an `Abs` directory.
absify :: Path a Dir -> Path Abs Dir
absify (Path.Internal.Path s) = mkInternalDir (T.pack s)


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


gitForceCommitAll :: Path Extant Dir -> String -> IO Repo
gitForceCommitAll root msg = do
  Git.runGit gitConfig $ do
    Git.initDB False >> Git.add ["."]
    Git.commit [] author email msg ["--allow-empty"]
  pure (Repo root gitConfig)
  where
    gitConfig = Git.makeConfig (toFilePath root) Nothing
    author    = "ki-author"
    email     = "ki-email"


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


gitIsDirty' :: Git.GitCtx Bool
gitIsDirty' = do
  staged   <- Git.gitExec "diff-index" ["--quiet", "--cached", "HEAD", "--"] []
  unstaged <- Git.gitExec "diff-files" ["--quiet"] []
  empty    <- Git.gitExec "rev-parse" ["HEAD"] []
  case (staged, unstaged, empty) of
    (Right _, Right _, _) -> pure False
    (_, _, Left _) -> pure False
    (_, _, _) -> pure True


gitIsDirty :: Repo -> IO Bool
gitIsDirty (Repo _ config) = Git.runGit config gitIsDirty'


-- ================================ Core logic ================================


gitCommitKiRepo :: Path Extant Dir -> String -> IO Repo
gitCommitKiRepo root msg = do
  Git.runGit gitConfig $ do
    Git.initDB False
    _ <- Git.gitExec "submodule" ["add", "./_media/"] []
    Git.add ["."]
    Git.commit [] author email msg []
  pure (Repo root gitConfig)
  where
    gitConfig = Git.makeConfig (toFilePath root) Nothing
    author    = "ki-author"
    email     = "ki-email"


-- | Convert a list of raw `SQLDeck`s into a preorder traversal of components.
-- This can be reversed to get a postorder traversal.
mkDeckTree :: [SQLDeck] -> [Deck]
mkDeckTree = L.sort . map unpack
  where
    unpack :: SQLDeck -> Deck
    unpack (SQLDeck did fullname) = Deck (T.splitOn "\x1f" fullname) (Did did)


mapFieldsByMid :: [FieldDef] -> Map Mid (Map FieldOrd FieldDef)
mapFieldsByMid = foldr go M.empty
  where
    go :: FieldDef -> Map Mid (Map FieldOrd FieldDef) -> Map Mid (Map FieldOrd FieldDef)
    go fld@(FieldDef mid ord _) fieldDefsByOrdByMid = if M.member mid fieldDefsByOrdByMid
      then M.adjust (M.insert ord fld) mid fieldDefsByOrdByMid
      else M.insert mid (M.singleton ord fld) fieldDefsByOrdByMid


mapTemplatesByMid :: [Template] -> Map Mid (Map TemplateOrd Template)
mapTemplatesByMid = foldr go M.empty
  where
    go :: Template -> Map Mid (Map TemplateOrd Template) -> Map Mid (Map TemplateOrd Template)
    go tmpl@(Template mid ord _) tmplsByOrdByMid = if M.member mid tmplsByOrdByMid
      then M.adjust (M.insert ord tmpl) mid tmplsByOrdByMid
      else M.insert mid (M.singleton ord tmpl) tmplsByOrdByMid


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
    htmlRegex = "<\\/?\\s*[a-z-][^>]*\\s*>|(\\&([\\w\\d]+|#\\d+|#x[a-f\\d]+);)" :: Text
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


getModel :: Map Mid (Map FieldOrd FieldDef)
         -> Map Mid (Map TemplateOrd Template)
         -> SQLModel
         -> Maybe Model
getModel fieldsByOrdByMid templatesByOrdByMid (SQLModel mid name _) =
  case (M.lookup (Mid mid) fieldsByOrdByMid, M.lookup (Mid mid) templatesByOrdByMid) of
    (Just fieldsByOrd, Just templatesByOrd) ->
      Just (Model (Mid mid) (ModelName name) fieldsByOrd templatesByOrd)
    _ -> Nothing


getField :: SQLField -> FieldDef
getField (SQLField mid ord name _) = FieldDef (Mid mid) (FieldOrd ord) (FieldName name)


getTemplate :: SQLTemplate -> Template
getTemplate (SQLTemplate mid ord name _) = Template (Mid mid) (TemplateOrd ord) (TemplateName name)

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
    -- It is important that we use `Integer` here and not `Int`, because
    -- otherwise, we get an overflow.
    b91Digits = mapM (\c -> fromIntegral <$> T.findIndex (== c) b91s) (T.unpack guid)
    val = L.foldl' go 0 <$> b91Digits

    go :: Integer -> Integer -> Integer
    go acc x = acc * 91 + x


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


mkSortField :: SQLData -> Maybe SortField
mkSortField (SQLInteger k) = Just $ SortField (T.pack $ show k)
mkSortField (SQLText t) = Just $ SortField t
mkSortField (SQLFloat x) = Just $ SortField (T.pack $ show x)
mkSortField (SQLBlob s) = Just $ SortField (T.pack $ show s)
mkSortField SQLNull = Just $ SortField ""


mkColNote :: Map Mid Model -> SQLNote -> Maybe ColNote
mkColNote modelsByMid (SQLNote nid mid guid tags flds sfld) = do
  (Model _ modelName fieldsByOrd _) <- M.lookup (Mid mid) modelsByMid
  fields    <- map mkField <$> (zipEq fs . L.sort . M.elems) fieldsByOrd
  sortField <- mkSortField sfld
  let mdNote = MdNote (Guid guid) modelName (Tags ts) (Fields fields) sortField
  pure (ColNote mdNote (Nid nid) (mkFilename mdNote))
  where
    ts = T.words tags
    fs = T.split (== '\x1f') flds
    mkField :: (Text, FieldDef) -> Field
    mkField (s, FieldDef _ ord name) = Field ord name s


getCard :: Map Nid ColNote -> [SQLDeck] -> SQLCard -> Maybe Card
getCard colnotesByNid decks (SQLCard cid nid did ord) = case (maybeColNote, maybeDeckName) of
  (Just colnote, Just deckName) ->
    Just $ Card (Cid cid) (Nid nid) (Did did) (Ord' ord) deckName colnote
  _ -> Nothing
  where
    unpack :: SQLDeck -> (Integer, Text)
    unpack (SQLDeck did' name) = (did', name)
    maybeColNote  = M.lookup (Nid nid) colnotesByNid
    maybeDeckName = DeckName <$> (M.lookup did . M.fromList . map unpack) decks




-- | This grabs the filename without the file extension, appends `.media`, and
-- then converts it back to a path.
--
-- We could use `takeBaseName` instead.
ankiMediaDirname :: Path Extant File -> Path Rel Dir
ankiMediaDirname colFile = mkInternalDir $ T.pack $ stem ++ ".media"
  where stem = (takeBaseName . toFilePath) colFile



-- | Append the md5sum of the collection file to the hashes file.
appendHash :: Path Extant Dir -> String -> MD5Digest -> IO ()
appendHash kiDir tag md5sum = do
  let hashesFile = kiDir </> Path.Internal.Path "hashes"
  appendFile (toFilePath hashesFile) (show md5sum ++ "  " ++ tag ++ "\n")


writeModel :: Path Extant Dir -> Model -> IO ()
writeModel modelsDir m@(Model _ modelName _ _) = LB.writeFile
  (toFilePath $ modelsDir </> Path.Internal.Path (show modelName ++ ".yaml"))
  (Y.encode [m])


mkNotePath :: Path Extant Dir -> Text -> Path Abs File
mkNotePath dir filename = absify dir </> (Path.Internal.Path . T.unpack) filename


mkDeckDir :: Path Extant Dir -> [Text] -> Path Abs Dir
mkDeckDir targetDir [] = absify targetDir
mkDeckDir targetDir (p : ps) =
  absify targetDir </> L.foldl' go (mkInternalDir p) ps
  where
    go :: Path Rel Dir -> Text -> Path Rel Dir
    go acc x = acc </> mkInternalDir x



mkPayload :: MdNote -> Text
mkPayload (MdNote (Guid guid) (ModelName modelName) (Tags tags) (Fields fields) _) = do
  header <> "\n" <> body
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
    body = L.foldl' go "" fields
    go :: Text -> Field -> Text
    go s (Field _ (FieldName name) text) =
      s <> "\n## " <> name <> "\n" <> (escapeMediaFilenames . htmlToScreen) text <> "\n"


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
    . subRegex' "\\<style\\>\\<\\/style\\>" ""


tagNameParser :: Parser Text
tagNameParser = do
  A.asciiCI "<" >> A.asciiCI "img" <|> A.asciiCI "audio" <|> A.asciiCI "object"


mediaAttrParser :: Parser Text
mediaAttrParser = do
  A.asciiCI "src=" <|> A.asciiCI "data="


-- | Parse the characters between the tag name and the `src`/`data` attribute.
--
-- This substring will always be nonempty, hence the `spacer`.
prefixAttrsParser :: Parser Text
prefixAttrsParser = do
  spacer <- A.notChar '>'
  T.cons spacer <$> (T.pack <$> A.manyTill (A.notChar '>') (lookAhead mediaAttrParser))


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


-- | Parse an HTML tag for some media file.
--
-- This parser expects the input stream to end immediately after the tag, thus
-- it is only suitable to be used with `streamEdit`.
mediaTagParser :: Parser MediaTag
mediaTagParser = do
  tagName <- tagNameParser
  prefixAttrs <- prefixAttrsParser
  attr    <- A.asciiCI "src=" <|> A.asciiCI "data="
  (MediaFilename fname pre suf) <-
    (dubQuotedFilenameParser <|> quotedFilenameParser <|> unquotedFilenameParser) <* A.char '>'
  A.endOfInput
  pure $ MediaTag fname ("<" <> tagName <> prefixAttrs <> attr <> pre) (suf <> ">")


-- | URI-decode all media filenames within HTML media tags.
escapeMediaFilenames :: Text -> Text
escapeMediaFilenames = streamEdit mediaTagParser mediaTagEditor


-- | URI-decode a media filename within some HTML media tag.
mediaTagEditor :: MediaTag -> Text
mediaTagEditor (MediaTag filename prefix suffix) = prefix <> URI.decodeText filename <> suffix


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
          -> ProgressBar ()
          -> Map Nid (Map Did (Path Extant File))
          -> Card
          -> IO (Map Nid (Map Did (Path Extant File)))
writeCard targetDir (Deck parts did) pb noteFilesByDidByNid (Card _ nid _ _ _ (ColNote mdnote _ stem))
  = do
    incProgress pb 1
    deckDir <- ensureExtantDir (mkDeckDir targetDir parts)
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
  where
    filename = stem <> ".md"
    payload  = mkPayload mdnote


writeDeck :: Path Extant Dir -> [Card] -> ProgressBar () -> Deck -> IO ()
writeDeck targetDir cards pb deck@(Deck _ did) = do
  foldM_ (writeCard targetDir deck pb) M.empty deckCards
  where deckCards = filter (\(Card _ _ did' _ _ _) -> did' == did) cards


writeRepo :: Path Extant File
          -> Path Extant Dir
          -> Path Extant Dir
          -> Path Extant Dir
          -> Path Extant Dir
          -> Path Extant Dir
          -> MD5Digest
          -> IO Repo
writeRepo colFile targetDir kiDir mediaDir ankiMediaDir modelsDir md5sum = do
  LB.writeFile (toFilePath $ kiDir </> Path.Internal.Path "config") $ JSON.encode config
  conn  <- SQL.open (toFilePath colFile)
  ns    <- SQL.query_ conn "SELECT id,mid,guid,tags,flds,sfld FROM notes" :: IO [SQLNote]
  cs    <- SQL.query_ conn "SELECT id,nid,did,ord FROM cards" :: IO [SQLCard]
  ds    <- SQL.query_ conn "SELECT id,name FROM decks" :: IO [SQLDeck]
  nts   <- SQL.query_ conn "SELECT id,name,config FROM notetypes" :: IO [SQLModel]
  flds  <- SQL.query_ conn "SELECT ntid,ord,name,config FROM fields" :: IO [SQLField]
  tmpls <- SQL.query_ conn "SELECT ntid,ord,name,config FROM templates" :: IO [SQLTemplate]
  let
    fieldsByMid = (mapFieldsByMid . map getField) flds
    templatesByMid = (mapTemplatesByMid . map getTemplate) tmpls
    maybeModels = map (getModel fieldsByMid templatesByMid) nts
    modelsByMid = M.fromList (MB.mapMaybe (fmap (\m@(Model mid _ _ _) -> (mid, m))) maybeModels)
    colnotesByNid = M.fromList $ MB.mapMaybe (fmap unpack . mkColNote modelsByMid) ns
    serializedModelsByMid = M.map (Y.encode . (: [])) modelsByMid
    cards     = MB.mapMaybe (getCard colnotesByNid ds) cs
    preorder  = mkDeckTree ds
    postorder = reverse preorder
  ankiMediaRepo <- gitForceCommitAll ankiMediaDir "Initial commit"
  printf "Cloning media from Anki media directory '%s'...\n" (toFilePath ankiMediaDir)
  _kiMediaRepo <- gitClone ankiMediaRepo mediaDir
  -- Dump all models to top-level `_models` subdirectory.
  forM_ (M.elems modelsByMid) (writeModel modelsDir)
  pb <- newProgressBar style 10 (Progress 0 (length cards) ())
  forM_ postorder (writeDeck targetDir cards pb)
  appendHash kiDir (toFilePath colFile) md5sum
  printf "Committing contents to repository...\n"
  repo <- gitCommitKiRepo targetDir "Initial commit"
  printf "Done!\n"
  pure repo
  where
    remote = JSON.Object $ M.singleton "path" $ (JSON.String . T.pack . toFilePath) colFile
    config = JSON.Object $ M.singleton "remote" remote
    style  = defStyle { styleWidth = ConstantWidth 72 }

    unpack :: ColNote -> (Nid, ColNote)
    unpack c@(ColNote _ nid _) = (nid, c)


continueClone :: Path Extant File -> Path Extant Dir -> IO ()
continueClone colFile targetDir = do
  -- Hash the collection file.
  colFileContents <- LB.readFile (toFilePath colFile)
  let colFileMD5 = md5 colFileContents
  -- Add the backups directory to the `.gitignore` file.
  writeFile (toFilePath gitIgnore) ".ki/backups\n"
  -- Create `.ki` and `_media` subdirectories.
  maybeKiDir     <- ensureEmpty (absify targetDir </> mkInternalDir ".ki/")
  maybeMediaDir  <- ensureEmpty (absify targetDir </> mkInternalDir "_media/")
  maybeModelsDir <- ensureEmpty (absify targetDir </> mkInternalDir "_models/")
  ankiMediaDir   <- ensureExtantDir (absify ankiUserDir </> ankiMediaDirname colFile)
  case (maybeKiDir, maybeMediaDir, maybeModelsDir) of
    (Nothing, _, _) -> printf "fatal: new '.ki' directory not empty\n"
    (_, Nothing, _) -> printf "fatal: new '_media' directory not empty\n"
    (_, _, Nothing) -> printf "fatal: new '_models' directory not empty\n"
    -- Write repository contents and commit.
    (Just kiDir, Just mediaDir, Just modelsDir) -> do
      repo    <- writeRepo colFile targetDir kiDir mediaDir ankiMediaDir modelsDir colFileMD5
      isDirty <- gitIsDirty repo
      when isDirty $ do
        printf "fatal: non-empty working tree in freshly cloned ki repo\n"
        exitFailure
  where
    gitIgnore   = absify targetDir </> Path.Internal.Path ".gitignore" :: Path Abs File
    ankiUserDir = parent colFile :: Path Extant Dir


-- Parse the collection and target directory, then call `continueClone`.
clone :: String -> String -> IO ()
clone colPath targetPath = do
  maybeColFile   <- resolveFile' colPath >>= getExtantFile
  maybeTargetDir <- resolveDir' targetPath >>= ensureEmpty
  case (maybeColFile, maybeTargetDir) of
    (Nothing, _) -> printf "fatal: collection file '%s' does not exist\n" colPath
    (_, Nothing) -> printf "fatal: targetdir '%s' not empty\n" targetPath
    (Just colFile, Just targetDir) -> continueClone colFile targetDir


main :: IO ()
main = do
  args <- getArgs
  case args of
    [col, tgt] -> clone col tgt
    _ -> putStrLn "Usage: ki <collection.anki2> <target>"
