{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
module Main (main) where

import Control.Monad (forM_)
import Data.ByteString.Lazy (ByteString)
import Data.Digest.Pure.MD5 (MD5Digest, md5)
import Data.Map (Map)
import Data.Text (Text)
import Data.Text.ICU.Replace (replaceAll)
import Data.Typeable (Typeable)
import Data.YAML ((.=), ToYAML(..))
import Path ((</>), Abs, Dir, File, Path, Rel, parent, toFilePath)
import Path.IO (doesFileExist, ensureDir, listDir, resolveDir', resolveFile')
import System.FilePath (takeBaseName)
import Text.Printf (printf)

import qualified Data.Aeson.Micro as JSON
import qualified Data.ByteString.Lazy as LB
import qualified Data.Map as M
import qualified Data.Maybe as MB
import qualified Data.Text as T
import qualified Data.Text.ICU as ICU
import qualified Data.YAML as Y
-- import qualified Data.ProtocolBuffers as PB
import qualified Database.SQLite.Simple as SQL
import qualified Lib.Git as Git
import qualified Path.Internal


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
  deriving Show

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
newtype Did = Did Integer
newtype Nid = Nid Integer deriving (Eq, Ord)
newtype Guid = Guid Text
newtype Tags = Tags [Text]
newtype Fields = Fields (Map FieldName Text)
newtype DeckName = DeckName Text
newtype SortField = SortField Text
newtype ModelName = ModelName Text deriving (ToYAML)

data Ord' = Ord' Integer

type Filename = Text
data MdNote = MdNote !Guid !ModelName !Tags !Fields !SortField
data ColNote = ColNote !MdNote !Nid !Filename
data Card = Card !Cid !Nid !Did !Ord' !DeckName

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

data Field = Field !Mid !FieldOrd !FieldName
data Template = Template !Mid !FieldOrd !TemplateName
data Model = Model
  { modelMid  :: !Mid
  , modelName :: !ModelName
  , modelFieldsAndTemplatesByOrd :: !(Map FieldOrd (Field, Template))
  }

instance ToYAML Field where
  toYAML (Field mid ord name) = Y.mapping ["mid" .= mid, "ord" .= ord, "name" .= name]

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
  return $ case contents of
    ([], []) -> Just $ Path.Internal.Path (toFilePath dir)
    _ -> Nothing


ensureExtantDir :: Path Abs Dir -> IO (Path Extant Dir)
ensureExtantDir dir = do
  ensureDir dir
  return $ Path.Internal.Path (toFilePath dir)


getExtantFile :: Path Abs File -> IO (Maybe (Path Extant File))
getExtantFile file = do
  exists <- doesFileExist file
  if exists then return $ Just $ Path.Internal.Path (toFilePath file) else return Nothing

-- | Convert from an `Extant` or `Missing` path to an `Abs` path.
absify :: Path a b -> Path Abs b
absify (Path.Internal.Path s) = Path.Internal.Path s

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
      Right _   -> return ()
      Left  err -> Git.gitError err "clone"
  pure $ Repo tgt $ Git.makeConfig (toFilePath tgt) Nothing


-- ================================ Core logic ================================


mapFieldsByMid :: [Field] -> Map Mid (Map FieldOrd Field)
mapFieldsByMid = foldr f M.empty
  where
    g :: FieldOrd -> Field -> Map FieldOrd Field -> Maybe (Map FieldOrd Field)
    g ord fld fieldsByOrd = Just $ M.insert ord fld fieldsByOrd
    f :: Field -> Map Mid (Map FieldOrd Field) -> Map Mid (Map FieldOrd Field)
    f fld@(Field mid ord _) = M.update (g ord fld) mid


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


plainToHtml :: Text -> Text
plainToHtml s = case ICU.find htmlRegex t of
  Nothing  -> T.replace "\n" "<br>" t
  (Just _) -> t
  where
    htmlRegex = "</?\\s*[a-z-][^>]*\\s*>|(\\&(?:[\\w\\d]+|#\\d+|#x[a-f\\d]+);)"
    sub :: Text -> Text
    sub =
      replaceAll "<div>\\s*</div>" ""
        . replaceAll "<i>\\s*</i>" ""
        . replaceAll "<b>\\s*</b>" ""
        . T.replace "&nbsp;" " "
        . T.replace "&amp;" "&"
        . T.replace "&gt;" ">"
        . T.replace "&lt;" "<"
    t = sub s


getModel :: Map Mid (Map FieldOrd (Field, Template)) -> SQLModel -> Maybe Model
getModel fieldsAndTemplatesByMid (SQLModel mid name _) =
  case M.lookup (Mid mid) fieldsAndTemplatesByMid of
    Just m  -> Just (Model (Mid mid) (ModelName name) m)
    Nothing -> Nothing


getFieldsAndTemplatesByMid :: Map Mid (Map FieldOrd Field)
                           -> Map Mid (Map FieldOrd Template)
                           -> Map Mid (Map FieldOrd (Field, Template))
getFieldsAndTemplatesByMid fieldsByMid templatesByMid = M.foldrWithKey f M.empty fieldsByMid
  where
    f :: Mid
      -> Map FieldOrd Field
      -> Map Mid (Map FieldOrd (Field, Template))
      -> Map Mid (Map FieldOrd (Field, Template))
    f mid fieldsByOrd acc = case M.lookup mid templatesByMid of
      Just templatesByOrd -> M.insert mid (M.intersectionWith (,) fieldsByOrd templatesByOrd) acc
      Nothing -> acc


getField :: SQLField -> Field
getField (SQLField mid ord name _) = Field (Mid mid) (FieldOrd ord) (FieldName name)


getTemplate :: SQLTemplate -> Template
getTemplate (SQLTemplate mid ord name _) = Template (Mid mid) (FieldOrd ord) (TemplateName name)


getFilename :: MdNote -> Text
getFilename (MdNote _ _ _ _ (SortField sfld)) =
  (T.pack . stripHtmlTags . T.unpack . plainToHtml) sfld


getFieldTextByFieldName :: [Text] -> Map FieldOrd Field -> Map FieldName Text
getFieldTextByFieldName fs m = M.fromList $ zip (map f (M.toAscList m)) fs
  where
    f :: (FieldOrd, Field) -> FieldName
    f (_, Field _ _ name) = name


getColNote :: Map Mid Model -> SQLNote -> Maybe ColNote
getColNote modelsByMid (SQLNote nid mid guid tags flds sfld) =
  ColNote <$> mdNote <*> Just (Nid nid) <*> (getFilename <$> mdNote)
  where
    ts = T.words tags
    fs = T.split (== '\x1f') flds
    maybeModel = M.lookup (Mid mid) modelsByMid
    maybeModelName = modelName <$> maybeModel
    maybeFieldsByOrd = M.map fst . modelFieldsAndTemplatesByOrd <$> maybeModel
    maybeFieldTextByFieldName = getFieldTextByFieldName fs <$> maybeFieldsByOrd
    mdNote =
      MdNote (Guid guid)
        <$> maybeModelName
        <*> Just (Tags ts)
        <*> (Fields <$> maybeFieldTextByFieldName)
        <*> Just (SortField sfld)


getCard :: [SQLDeck] -> SQLCard -> Maybe Card
getCard decks (SQLCard cid nid did ord) =
  Card (Cid cid) (Nid nid) (Did did) (Ord' ord)
    <$> DeckName
    <$> (M.lookup did . M.fromList . map unpack) decks
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
  return ()


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
    fieldsByMid    = (mapFieldsByMid . map getField) flds
    templatesByMid = (mapTemplatesByMid . map getTemplate) tmpls
    fieldsAndTemplatesByMid = getFieldsAndTemplatesByMid fieldsByMid templatesByMid
    modelsByMid =
      M.fromList $ MB.mapMaybe (fmap (\m -> (modelMid m, m)) . getModel fieldsAndTemplatesByMid) nts
    colnotesByNid = M.fromList $ MB.mapMaybe (fmap unpack . getColNote modelsByMid) ns
    serializedModelsByMid = M.map (Y.encode . (: [])) modelsByMid
    cards = (MB.catMaybes . map (getCard ds)) cs
  ankiMediaRepo <- gitCommitAll ankiMediaDir "Initial commit."
  _kiMediaRepo  <- gitClone ankiMediaRepo mediaDir
  -- Dump all models to top-level `_models` subdirectory.
  forM_ (M.assocs serializedModelsByMid) (writeModel modelsDir)
  writeDecks targetDir colnotesByNid cards
  where
    remote = JSON.Object $ M.singleton "path" $ (JSON.String . T.pack . toFilePath) colFile
    config = JSON.Object $ M.singleton "remote" remote

    unpack :: ColNote -> (Nid, ColNote)
    unpack c@(ColNote _ nid _) = (nid, c)


writeModel :: Path Extant Dir -> (Mid, ByteString) -> IO ()
writeModel modelsDir (mid, s) =
  LB.writeFile (toFilePath $ modelsDir </> Path.Internal.Path (show mid ++ ".yaml")) s


writeDecks :: Path Extant Dir -> Map Nid ColNote -> [Card] -> IO ()
writeDecks _targetDir _colnotesByNid _cards = do
  return ()


main :: IO ()
main = do
  print ("Hello" :: String)
  clone "" ""
