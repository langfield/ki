{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
module Ki where

import Control.Applicative ((<|>), many)
import Control.Monad (foldM_, forM_, when)
import Data.Attoparsec.Combinator (lookAhead)
import Data.Attoparsec.Text (Parser)
import Data.ByteString.Lazy (ByteString)
import Data.Digest.Pure.MD5 (MD5Digest, md5)
import Data.Map (Map)
import Data.ProtoLens.Encoding (decodeMessage)
import Data.Text (Text)
import Data.Text.Encoding (encodeUtf8)
import Data.Typeable (Typeable)
import Data.YAML ((.=), ToYAML(..))
import Database.SQLite.Simple (SQLData(..))
import Lens.Micro ((^.))
import Numeric (showHex)
import Path ((</>), Abs, Dir, File, Path, Rel, parent, toFilePath)
import Path.IO (createFileLink, doesFileExist, ensureDir, listDir, resolveDir', resolveFile')
import Replace.Attoparsec.Text (streamEdit)
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
import qualified Data.YAML.Event as Y
import qualified Data.YAML.Schema as Y
import qualified Data.YAML.Token as Y
import qualified Database.SQLite.Simple as SQL
import qualified Lib.Git as GIT
import qualified Network.URI.Encode as URI
import qualified Path.Internal
import qualified Proto.Notetypes as ANKI
import qualified Proto.Notetypes_Fields as ANKI

-- Upper bound on the length of Ki-generated card filenames.
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

-- Types prefixed with `SQL` are product types designed to hold the raw data
-- parsed from the `.anki2` SQLite3 database dump binary. Each of these is then
-- parsed into a more structured twin. For example, `SQLNote`s are parsed into
-- `ColNote`s via `mkColNote`.

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

-- Anki core types.
--
-- N.B.: `MdNote` stands for "markdown note", and is a representation of the
-- data that is dumped to a card file in a ki repository.
--
-- A `ColNote` ("collection note") is a wrapper around a markdown note that
-- includes the `Nid` (note ID), and the filename.

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

data NotetypeKind = Normal | Cloze deriving (Eq, Show)
data CardKind = None | Any | All deriving (Eq, Show)
data CardRequirement = CardRequirement
  { cardOrd  :: Integer
  , cardKind :: CardKind
  , cardFieldOrds :: [Integer]
  }
  deriving (Eq, Show)

-- Wrappers around `proto-lens`-generated types.
newtype NotetypeConfig = NotetypeConfig ANKI.Notetype'Config deriving (Eq, Show)
newtype NotetypeConfigKind = NotetypeConfigKind ANKI.Notetype'Config'Kind deriving (Eq, Show)
newtype NotetypeConfigCardRequirementKind = NotetypeConfigCardRequirementKind ANKI.Notetype'Config'CardRequirement'Kind deriving (Eq, Show)
newtype NotetypeConfigCardRequirement = NotetypeConfigCardRequirement ANKI.Notetype'Config'CardRequirement deriving (Eq, Show)
newtype FieldDefConfig = FieldDefConfig ANKI.Notetype'Field'Config deriving (Eq, Show)
newtype TemplateConfig = TemplateConfig ANKI.Notetype'Template'Config deriving (Eq, Show)

-- ToYAML instances for serializing models (notetypes).

instance ToYAML FieldDefConfig where
  toYAML (FieldDefConfig config) = Y.mapping
    [ "sticky" .= (config ^. ANKI.sticky)
    , "rtl" .= (config ^. ANKI.rtl)
    , "font_name" .= (config ^. ANKI.fontName)
    , "font_size" .= (config ^. ANKI.fontSize)
    , "description" .= (config ^. ANKI.description)
    , "plain_text" .= (config ^. ANKI.plainText)
    , "collapsed" .= (config ^. ANKI.collapsed)
    , "other" .= (T.pack . LB.foldr showHex "" . LB.fromStrict) (config ^. ANKI.other)
    ]

instance ToYAML NotetypeConfigKind where
  toYAML (NotetypeConfigKind ANKI.Notetype'Config'KIND_NORMAL) = Y.toYAML ("normal" :: Text)
  toYAML (NotetypeConfigKind ANKI.Notetype'Config'KIND_CLOZE) = Y.toYAML ("cloze" :: Text)
  toYAML _ = Y.toYAML ("unknown" :: Text)

instance ToYAML NotetypeConfigCardRequirementKind where
  toYAML (NotetypeConfigCardRequirementKind ANKI.Notetype'Config'CardRequirement'KIND_NONE) =
    Y.toYAML ("none" :: Text)
  toYAML (NotetypeConfigCardRequirementKind ANKI.Notetype'Config'CardRequirement'KIND_ANY) =
    Y.toYAML ("any" :: Text)
  toYAML (NotetypeConfigCardRequirementKind ANKI.Notetype'Config'CardRequirement'KIND_ALL) =
    Y.toYAML ("all" :: Text)
  toYAML _ = Y.toYAML ("unknown" :: Text)

instance ToYAML NotetypeConfigCardRequirement where
  toYAML (NotetypeConfigCardRequirement req) = Y.mapping
    [ "kind" .= NotetypeConfigCardRequirementKind (req ^. ANKI.kind)
    , "card_ord" .= (req ^. ANKI.cardOrd)
    , "field_ords" .= (req ^. ANKI.fieldOrds)
    ]

instance ToYAML NotetypeConfig where
  toYAML (NotetypeConfig config) = Y.mapping
    [ "kind" .= NotetypeConfigKind (config ^. ANKI.kind)
    , "sort_field" .= (config ^. ANKI.sortFieldIdx)
    , "css" .= (config ^. ANKI.css)
    , "latex_preamble" .= (config ^. ANKI.latexPre)
    , "latex_postamble" .= (config ^. ANKI.latexPost)
    , "latex_use_SVGs" .= (config ^. ANKI.latexSvg)
    , "card_reqs" .= map NotetypeConfigCardRequirement (config ^. ANKI.reqs)
    , "other" .= (T.pack . LB.foldr showHex "" . LB.fromStrict) (config ^. ANKI.other)
    ]

instance ToYAML TemplateConfig where
  toYAML (TemplateConfig config) = Y.mapping
    [ "question_format" .= (config ^. ANKI.qFormat)
    , "answer_format" .= (config ^. ANKI.aFormat)
    , "question_format_browser" .= (config ^. ANKI.qFormatBrowser)
    , "answer_format_browser" .= (config ^. ANKI.aFormatBrowser)
    , "target_deck_id" .= (config ^. ANKI.targetDeckId)
    , "browser_font_name" .= (config ^. ANKI.browserFontName)
    , "browser_font_size" .= (config ^. ANKI.browserFontSize)
    , "other" .= (T.pack . LB.foldr showHex "" . LB.fromStrict) (config ^. ANKI.other)
    ]

data FieldDef = FieldDef !Mid !FieldOrd !FieldName !FieldDefConfig
  deriving (Eq, Show)
data Template = Template !Mid !TemplateOrd !TemplateName !TemplateConfig
  deriving Show
data Model = Model
  !Mid
  !ModelName
  !(Map FieldOrd FieldDef)
  !(Map TemplateOrd Template)
  !NotetypeConfig
  deriving Show

instance Ord FieldDef where
  (FieldDef _ (FieldOrd ord) _ _) <= (FieldDef _ (FieldOrd ord') _ _) = ord <= ord'

instance ToYAML FieldDef where
  toYAML (FieldDef mid ord name config) =
    Y.mapping ["mid" .= mid, "ord" .= ord, "name" .= name, "config" .= config]

instance ToYAML Template where
  toYAML (Template mid ord name config) =
    Y.mapping ["mid" .= mid, "ord" .= ord, "name" .= name, "config" .= config]

instance ToYAML Model where
  toYAML (Model mid ord fieldsByOrd templatesByOrd config) = Y.mapping
    [ "mid" .= mid
    , "ord" .= ord
    , "fieldsByOrd" .= fieldsByOrd
    , "templatesByOrd" .= templatesByOrd
    , "config" .= config
    ]

data Repo = Repo !(Path Extant Dir) !GIT.Config

type Prefix = Text
type Suffix = Text
data MediaFilename = MediaFilename !Filename !Prefix !Suffix
  deriving Show
data MediaTag = MediaTag !Filename !Prefix !Suffix
  deriving Show

-- ========================== Path utility functions ==========================

-- | Cast a path name to a directory (absolute or relative, stripping pathseps).
getDir :: Text -> Path a Dir
getDir s = Path.Internal.Path (T.unpack withSlash)
  where
    stripped  = subRegexText "\\/+$" "" s
    withSlash = if T.null stripped then "./" else stripped <> "/"

-- | Cast a directory to absolute or relative.
castDir :: Path b Dir -> Path b' Dir
castDir = getDir . T.pack . toFilePath

-- | Ensure a directory is empty, creating it if necessary, and returning
-- `Nothing` if it was not.
ensureEmpty :: Path Abs Dir -> IO (Maybe (Path Extant Dir))
ensureEmpty dir = do
  ensureDir dir
  contents <- listDir dir
  pure $ if contents == ([], []) then Just $ castDir dir else Nothing

-- | Ensure a directory exists, creating it if necessary, and casting to `Extant`.
ensureExtantDir :: Path Abs Dir -> IO (Path Extant Dir)
ensureExtantDir dir = do
  ensureDir dir
  pure $ castDir dir

-- | Check if a file exists, returning a `Maybe`.
getExtantFile :: Path Abs File -> IO (Maybe (Path Extant File))
getExtantFile file = do
  exists <- doesFileExist file
  if exists then pure $ Just $ Path.Internal.Path (toFilePath file) else pure Nothing

-- | Convert from an `Extant` or `Missing` directory to an `Abs` directory.
absify :: Path a Dir -> Path Abs Dir
absify (Path.Internal.Path s) = getDir (T.pack s)

-- | Create a symlink to the first argument at the second argument, returning the link.
mkFileLink :: Path Extant File -> Path Abs File -> IO (Path Extant File)
mkFileLink tgt lnk = do
  createFileLink tgt lnk
  pure $ ((Path.Internal.Path . T.unpack) . T.pack . toFilePath) lnk

-- | Create a file with contents in some extant directory (overwrites).
mkNewFile :: Path Extant Dir -> Text -> Text -> IO (Path Extant File)
mkNewFile dir name contents = do
  writeFile (toFilePath file) (T.unpack contents)
  pure file
  where file = dir </> (Path.Internal.Path . T.unpack) name

-- ======================= Repository utility functions =======================

-- | Commit in a directory with a given message even if there are no staged changes.
gitForceCommitAll :: Path Extant Dir -> String -> IO Repo
gitForceCommitAll root msg = do
  GIT.runGit gitConfig $ do
    GIT.initDB False >> GIT.add ["."]
    GIT.commit [] author email msg ["--allow-empty"]
  pure (Repo root gitConfig)
  where
    gitConfig = GIT.makeConfig (toFilePath root) Nothing
    author    = "ki-author"
    email     = "ki-email"

-- | Clone a local repository into the given directory, which must exist and be
-- empty (data invariant).
gitClone :: Repo -> Path Extant Dir -> IO Repo
gitClone (Repo root config) tgt = do
  GIT.runGit config $ do
    o <- GIT.gitExec "clone" [toFilePath root, toFilePath tgt] []
    case o of
      Right _   -> pure ()
      Left  err -> GIT.gitError err "clone"
  pure $ Repo tgt $ GIT.makeConfig (toFilePath tgt) Nothing

-- | Check if the stage or working directory is dirty (libgit function).
gitIsDirty' :: GIT.GitCtx Bool
gitIsDirty' = do
  staged   <- GIT.gitExec "diff-index" ["--quiet", "--cached", "HEAD", "--"] []
  unstaged <- GIT.gitExec "diff-files" ["--quiet"] []
  empty    <- GIT.gitExec "rev-parse" ["HEAD"] []
  case (staged, unstaged, empty) of
    (Right _, Right _, _) -> pure False
    (_, _, Left _) -> pure False
    (_, _, _) -> pure True

-- | Check if the stage or working directory is dirty.
gitIsDirty :: Repo -> IO Bool
gitIsDirty (Repo _ config) = GIT.runGit config gitIsDirty'

-- ================================ Core logic ================================

-- | Commit in a ki repository with some message, adding `_media/` subdirectory
-- as a submodule.
gitCommitKiRepo :: Path Extant Dir -> String -> IO Repo
gitCommitKiRepo root msg = do
  GIT.runGit gitConfig $ do
    GIT.initDB False
    _ <- GIT.gitExec "submodule" ["add", "./_media/"] []
    GIT.add ["."]
    GIT.commit [] author email msg []
  pure (Repo root gitConfig)
  where
    gitConfig = GIT.makeConfig (toFilePath root) Nothing
    author    = "ki-author"
    email     = "ki-email"

-- | Convert a list of raw `SQLDeck`s into a preorder traversal of components.
-- This can be reversed to get a postorder traversal.
mkDeckTree :: [SQLDeck] -> [Deck]
mkDeckTree = L.sort . map unpack
  where
    unpack :: SQLDeck -> Deck
    unpack (SQLDeck did fullname) = Deck (T.splitOn "\x1f" fullname) (Did did)

-- | Associate each model ID with a map of field ordinals to field definitions.
--
-- * Each model has a set of fields, so this is the outer map.
-- * Field definitions are numbered (ordinals), so the inner map sends field
-- ordinals to the field definitions themselves.
mapFieldDefsByMid :: [FieldDef] -> Map Mid (Map FieldOrd FieldDef)
mapFieldDefsByMid = foldr go M.empty
  where
    go :: FieldDef -> Map Mid (Map FieldOrd FieldDef) -> Map Mid (Map FieldOrd FieldDef)
    go fld@(FieldDef mid ord _ _) fieldDefsByOrdByMid = if M.member mid fieldDefsByOrdByMid
      then M.adjust (M.insert ord fld) mid fieldDefsByOrdByMid
      else M.insert mid (M.singleton ord fld) fieldDefsByOrdByMid

-- | Associate each model ID with a map of template ordinals to template definitions.
--
-- * Each model has a set of templates, so this is the outer map.
-- * Templates are numbered (ordinals), so the inner map sends template
-- ordinals to the template definitions themselves.
mapTemplatesByMid :: [Template] -> Map Mid (Map TemplateOrd Template)
mapTemplatesByMid = foldr go M.empty
  where
    go :: Template -> Map Mid (Map TemplateOrd Template) -> Map Mid (Map TemplateOrd Template)
    go tmpl@(Template mid ord _ _) tmplsByOrdByMid = if M.member mid tmplsByOrdByMid
      then M.adjust (M.insert ord tmpl) mid tmplsByOrdByMid
      else M.insert mid (M.singleton ord tmpl) tmplsByOrdByMid

-- | Drop all characters within (and including) '<', '>' characters.
stripHtmlTags :: String -> String
stripHtmlTags "" = ""
stripHtmlTags ('<' : xs) = stripHtmlTags $ drop 1 $ dropWhile (/= '>') xs
stripHtmlTags (x : xs) = x : stripHtmlTags xs

-- | A wrapper around `subRegex` that works on `Text` instead of `String`.
subRegexText :: Text -> Text -> Text -> Text
subRegexText pat rep t = T.pack $ subRegex (mkRegex pat') t' rep'
  where
    pat' = T.unpack pat
    rep' = T.unpack rep
    t'   = T.unpack t

-- | Convert a plaintext representation of an anki note field to HTML (lossy).
plainToHtml :: Text -> Text
plainToHtml s
  | t =~ htmlRegex = T.replace "\n" "<br>" t
  | otherwise = t
  where
    htmlRegex = "<\\/?\\s*[a-z-][^>]*\\s*>|(\\&([\\w\\d]+|#\\d+|#x[a-f\\d]+);)" :: Text
    sub :: Text -> Text
    sub =
      subRegexText "<div>\\s*</div>" ""
        . subRegexText "<i>\\s*</i>" ""
        . subRegexText "<b>\\s*</b>" ""
        . T.replace "&nbsp;" " "
        . T.replace "&amp;" "&"
        . T.replace "&gt;" ">"
        . T.replace "&lt;" "<"
    t = sub s

-- | Convert data for a model (notetype) from generic SQL types into a
-- domain-specific representation.
mkModel :: Map Mid (Map FieldOrd FieldDef)
        -> Map Mid (Map TemplateOrd Template)
        -> SQLModel
        -> Maybe Model
mkModel fieldsByOrdByMid templatesByOrdByMid (SQLModel mid name configBytes) =
  case
      ( M.lookup (Mid mid) fieldsByOrdByMid
      , M.lookup (Mid mid) templatesByOrdByMid
      , decodeMessage (LB.toStrict configBytes) :: Either String ANKI.Notetype'Config
      )
    of
      (Just fieldsByOrd, Just templatesByOrd, Right config) ->
        Just $ Model (Mid mid) (ModelName name) fieldsByOrd templatesByOrd (NotetypeConfig config)
      _ -> Nothing

-- | Convert data for a field definition from generic SQL types into a
-- domain-specific representation.
mkFieldDef :: SQLField -> Maybe FieldDef
mkFieldDef (SQLField mid ord name bytes) =
  case decodeMessage (LB.toStrict bytes) :: Either String ANKI.Notetype'Field'Config of
    Right config ->
      Just $ FieldDef (Mid mid) (FieldOrd ord) (FieldName name) (FieldDefConfig config)
    _ -> Nothing

-- | Convert data for a card template definition from generic SQL types into a
-- domain-specific representation.
mkTemplate :: SQLTemplate -> Maybe Template
mkTemplate (SQLTemplate mid ord name bytes) =
  case decodeMessage (LB.toStrict bytes) :: Either String ANKI.Notetype'Template'Config of
    Right config ->
      Just $ Template (Mid mid) (TemplateOrd ord) (TemplateName name) (TemplateConfig config)
    _ -> Nothing

-- | Convert the raw contents of a field into an appropriate filename.
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

-- | Zip two lists, returning `Nothing` if they don't have the same length.
zipEq :: [a] -> [b] -> Maybe [(a, b)]
zipEq [] [] = Just []
zipEq [] _  = Nothing
zipEq _  [] = Nothing
zipEq (x : xs) (y : ys) = ((x, y) :) <$> zipEq xs ys

-- | Parse the sort field of a note (could be several different types).
mkSortField :: SQLData -> Maybe SortField
mkSortField (SQLInteger k) = Just $ SortField (T.pack $ show k)
mkSortField (SQLText t) = Just $ SortField t
mkSortField (SQLFloat x) = Just $ SortField (T.pack $ show x)
mkSortField (SQLBlob s) = Just $ SortField (T.pack $ show s)
mkSortField SQLNull = Just $ SortField ""

-- | Convert a `SQLNote` (an anki note expressed via the SQLite3 type system)
-- into a `ColNote` (an anki note expressed via our custom Haskell types).
mkColNote :: Map Mid Model -> SQLNote -> Maybe ColNote
mkColNote modelsByMid (SQLNote nid mid guid tags flds sfld) = do
  (Model _ modelName fieldsByOrd _ _) <- M.lookup (Mid mid) modelsByMid
  fields    <- map mkField <$> (zipEq fs . L.sort . M.elems) fieldsByOrd
  sortField <- mkSortField sfld
  let mdNote = MdNote (Guid guid) modelName (Tags ts) (Fields fields) sortField
  pure (ColNote mdNote (Nid nid) (mkFilename mdNote))
  where
    ts = T.words tags
    fs = T.split (== '\x1f') flds
    mkField :: (Text, FieldDef) -> Field
    mkField (s, FieldDef _ ord name _) = Field ord name s

-- | Convert a `SQLCard` (an anki card expressed via the SQLite3 type system)
-- into a `Card` (an anki card expressed via our custom Haskell types).
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
ankiMediaDirname colFile = getDir $ T.pack $ stem ++ ".media"
  where stem = (takeBaseName . toFilePath) colFile

-- | Append the md5sum of the collection file to the hashes file.
appendHash :: Path Extant Dir -> String -> MD5Digest -> IO ()
appendHash kiDir tag md5sum = do
  let hashesFile = kiDir </> Path.Internal.Path "hashes"
  appendFile (toFilePath hashesFile) (show md5sum ++ "  " ++ tag ++ "\n")

-- | A YAML helper function that ensures we use multiline strings.
useBlockStrings :: Y.Scalar -> Either String (Y.Tag, Y.ScalarStyle, Text)
useBlockStrings (Y.SStr s)
  | '\n' `elem` T.unpack s = Right (Y.untagged, Y.Literal Y.Keep Y.IndentAuto, s)
  | otherwise = Y.schemaEncoderScalar Y.coreSchemaEncoder (Y.SStr s)
useBlockStrings scalar = Y.schemaEncoderScalar Y.coreSchemaEncoder scalar

-- | Dump a model to disk in the given extant directory.
writeModel :: Path Extant Dir -> Model -> IO ()
writeModel modelsDir m@(Model _ modelName _ _ _) = LB.writeFile path payload
  where
    schemaEncoder = Y.setScalarStyle useBlockStrings Y.coreSchemaEncoder
    payload = Y.encodeNode' schemaEncoder Y.UTF8 (map (Y.Doc . Y.toYAML) [m])
    path    = toFilePath $ modelsDir </> Path.Internal.Path (show modelName ++ ".yaml")

-- | Get an absolute path given a directory and a filename.
mkNotePath :: Path Extant Dir -> Text -> Path Abs File
mkNotePath dir filename = absify dir </> (Path.Internal.Path . T.unpack) filename

-- | Given the 'parts' of a full deck name, construct the absolute path to that deck.
mkDeckDir :: Path Extant Dir -> [Text] -> Path Abs Dir
mkDeckDir targetDir [] = absify targetDir
mkDeckDir targetDir (p : ps) = absify targetDir </> L.foldl' go (getDir p) ps
  where
    go :: Path Rel Dir -> Text -> Path Rel Dir
    go acc x = acc </> getDir x

-- | Serialize a markdown note into plaintext, ready to be written to a file.
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

-- | Convert HTML into plaintext.
htmlToScreen :: Text -> Text
htmlToScreen =
  T.strip
    . subRegexText "\\<b\\>\\s*\\<\\/b\\>" ""
    . subRegexText "src= ?\n\"" "src=\""
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
    . subRegexText "\\<style\\>\\<\\/style\\>" ""

-- ============================ Attoparsec parsers ===========================

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

-- | Write all the cards for a given deck.
writeDeck :: Path Extant Dir -> [Card] -> ProgressBar () -> Deck -> IO ()
writeDeck targetDir cards pb deck@(Deck _ did) = do
  foldM_ (writeCard targetDir deck pb) M.empty deckCards
  where deckCards = filter (\(Card _ _ did' _ _ _) -> did' == did) cards

-- | Write files to an empty, initialized ki repository.
writeRepo :: Path Extant File
          -> Path Extant Dir
          -> Path Extant Dir
          -> Path Extant Dir
          -> Path Extant Dir
          -> Path Extant Dir
          -> MD5Digest
          -> IO Repo
writeRepo colFile targetDir kiDir mediaDir ankiMediaDir modelsDir md5sum = do
  -- Dump location of `.anki2` file to `.ki/config`.
  LB.writeFile (toFilePath $ kiDir </> Path.Internal.Path "config") $ JSON.encode config
  -- Read all the relevant SQL tables into memory.
  conn  <- SQL.open (toFilePath colFile)
  ns    <- SQL.query_ conn "SELECT id,mid,guid,tags,flds,sfld FROM notes" :: IO [SQLNote]
  cs    <- SQL.query_ conn "SELECT id,nid,did,ord FROM cards" :: IO [SQLCard]
  ds    <- SQL.query_ conn "SELECT id,name FROM decks" :: IO [SQLDeck]
  nts   <- SQL.query_ conn "SELECT id,name,config FROM notetypes" :: IO [SQLModel]
  flds  <- SQL.query_ conn "SELECT ntid,ord,name,config FROM fields" :: IO [SQLField]
  tmpls <- SQL.query_ conn "SELECT ntid,ord,name,config FROM templates" :: IO [SQLTemplate]
  -- Parse and preprocess all the raw data from the SQL tables.
  let
    fieldDefsByMid = (mapFieldDefsByMid . MB.mapMaybe mkFieldDef) flds
    templatesByMid = (mapTemplatesByMid . MB.mapMaybe mkTemplate) tmpls
    maybeModels = map (mkModel fieldDefsByMid templatesByMid) nts
    modelsByMid = M.fromList (MB.mapMaybe (fmap (\m@(Model mid _ _ _ _) -> (mid, m))) maybeModels)
    colnotesByNid = M.fromList $ MB.mapMaybe (fmap unpack . mkColNote modelsByMid) ns
    cards     = MB.mapMaybe (getCard colnotesByNid ds) cs
    preorder  = mkDeckTree ds
    postorder = reverse preorder
  -- Commit all files in `collection.media/` (remote), and clone into `_media/`.
  ankiMediaRepo <- gitForceCommitAll ankiMediaDir "Initial commit"
  printf "Cloning media from Anki media directory '%s'...\n" (toFilePath ankiMediaDir)
  _kiMediaRepo <- gitClone ankiMediaRepo mediaDir
  -- Dump all models to top-level `_models` subdirectory.
  forM_ (M.elems modelsByMid) (writeModel modelsDir)
  -- Serialize the cards for each deck.
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

-- | Clone an extant anki collection file into an extant target directory.
continueClone :: Path Extant File -> Path Extant Dir -> IO ()
continueClone colFile targetDir = do
  -- Hash the collection file.
  colFileContents <- LB.readFile (toFilePath colFile)
  let colFileMD5 = md5 colFileContents
  -- Add the backups directory to the `.gitignore` file.
  writeFile (toFilePath gitIgnore) ".ki/backups\n"
  -- Create `.ki` and `_media` subdirectories.
  maybeKiDir     <- ensureEmpty (absify targetDir </> getDir ".ki/")
  maybeMediaDir  <- ensureEmpty (absify targetDir </> getDir "_media/")
  maybeModelsDir <- ensureEmpty (absify targetDir </> getDir "_models/")
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

-- | Parse the collection and target directory, then call `continueClone`.
clone :: String -> String -> IO ()
clone colPath targetPath = do
  maybeColFile   <- resolveFile' colPath >>= getExtantFile
  maybeTargetDir <- resolveDir' targetPath >>= ensureEmpty
  case (maybeColFile, maybeTargetDir) of
    (Nothing, _) -> printf "fatal: collection file '%s' does not exist\n" colPath
    (_, Nothing) -> printf "fatal: targetdir '%s' not empty\n" targetPath
    (Just colFile, Just targetDir) -> continueClone colFile targetDir
