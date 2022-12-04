{-# LANGUAGE OverloadedStrings #-}
module Main (main) where

import Git (repoPath, RepositoryOptions, defaultRepositoryOptions)
import Git.Libgit2 (openLgRepository)
import Path (Path, Abs, Rel, File, Dir, toFilePath, (</>))
import Path.IO (resolveFile', resolveDir', ensureDir, listDir, doesFileExist)
import Text.Printf (printf)
import Data.Map (Map)
import Data.Text (Text)
import Data.Digest.Pure.MD5 (md5, MD5Digest)
import Data.Typeable (Typeable)
import Control.Monad.IO.Class (MonadIO)
import Database.SQLite.Simple.FromField (FromField)

import qualified Path.Internal
import qualified Data.Map as M
import qualified Data.Text as T
import qualified Data.Aeson.Micro as JSON
import qualified Data.ByteString.Lazy as LB
import qualified Database.SQLite.Simple as SQL


class AbsIO a

-- Extra base types for `Path` to distinguish between paths that exist (i.e.
-- there is a file or directory there) and paths that do not.
data Extant deriving (Typeable)
data Missing deriving (Typeable)

instance AbsIO Extant
instance AbsIO Missing

-- Extra type types for `Path`.
data Leaf deriving (Typeable)
data Link deriving (Typeable)
data Pseudo deriving (Typeable)
data WindowsLink deriving (Typeable)

-- Mid, Guid, Tags, Flds, SortFld
data SQLNote = SQLNote Integer Text Text Text Text deriving Show
-- Ntid, Name, Config (ProtoBuf message).
data SQLNotetype = SQLNotetype Integer Text Text deriving Show

newtype Guid = Guid Text
newtype Tags = Tags [Text]
newtype Model = Model Text
newtype Fields = Fields (Map Text Text)
newtype SortField = SortField Text

data MdNote = ColNote Guid Tags Model Fields SortField

instance SQL.FromRow SQLNote where
  fromRow = SQLNote <$> SQL.field <*> SQL.field <*> SQL.field <*> SQL.field <*> SQL.field

instance SQL.FromRow SQLNotetype where
  fromRow = SQLNotetype <$> SQL.field <*> SQL.field <*> SQL.field


-- Ensure a directory is empty, creating it if necessary, and returning
-- `Nothing` if it was not.
ensureEmpty :: Path Abs Dir -> IO (Maybe (Path Extant Dir))
ensureEmpty dir = do
  ensureDir dir
  contents <- listDir dir
  return $
    case contents of
      ([], []) -> Just $ Path.Internal.Path (toFilePath dir)
      _        -> Nothing


-- Ensure a file exists, creating it if necessary.
ensureFile :: Path Abs File -> IO (Path Extant File)
ensureFile file = do
  exists <- doesFileExist file
  case exists of
    True -> return $ Path.Internal.Path (toFilePath file)
    False -> do
      appendFile (toFilePath file) ""
      return (Path.Internal.Path (toFilePath file))


getExtantFile :: Path Abs File -> IO (Maybe (Path Extant File))
getExtantFile file = do
  exists <- doesFileExist file
  case exists of
    True -> return $ Just $ Path.Internal.Path (toFilePath file)
    False -> return Nothing


-- Convert from an `Extant` or `Missing` path to an `Abs` path.
absify :: AbsIO a => Path a b -> Path Abs b
absify (Path.Internal.Path s) = Path.Internal.Path s


-- Parse the collection and target directory, then call `continueClone`.
clone :: String -> String -> IO ()
clone colPath targetPath = do
  colFile <- resolveFile' colPath
  targetDir <- resolveDir' targetPath
  maybeColFile <- getExtantFile colFile
  maybeTargetDir <- ensureEmpty targetDir
  case (maybeColFile, maybeTargetDir) of
    (Nothing, _) -> printf "fatal: collection file '%s' does not exist" (show colFile)
    (_, Nothing) -> printf "fatal: targetdir '%s' not empty" (show targetDir)
    (Just colFile, Just targetDir) -> continueClone colFile targetDir


writeRepo :: Path Extant File -> Path Extant Dir -> Path Extant Dir -> Path Extant Dir -> IO [String]
writeRepo colFile targetDir kiDir mediaDir = do
  LB.writeFile (toFilePath $ kiDir </> Path.Internal.Path "config") $ JSON.encode configMap
  conn <- SQL.open (toFilePath colFile)
  notes <- SQL.query_ conn "SELECT (guid,mid,tags,flds,sfld) FROM notes" :: IO [SQLNote]
  models <- SQL.query_ conn "SELECT (id,name,config) FROM notetypes" :: IO [SQLNotetype]
  return [""]
  where
    remoteMap = JSON.Object $ M.singleton "path" $ (JSON.String . T.pack . toFilePath) colFile
    configMap = JSON.Object $ M.singleton "remote" $ remoteMap


continueClone :: Path Extant File -> Path Extant Dir -> IO ()
continueClone colFile targetDir = do
  -- Hash the collection file.
  colFileContents <- LB.readFile (toFilePath colFile)
  let colFileMD5 = md5 colFileContents
  -- Add the backups directory to the `.gitignore` file.
  writeFile (toFilePath gitIgnore) ".ki/backups"
  -- Create `.ki` and `_media` subdirectories.
  maybeKiDir <- ensureEmpty (absify targetDir </> Path.Internal.Path ".ki")
  maybeMediaDir <- ensureEmpty (absify targetDir </> Path.Internal.Path "_media")
  case (maybeKiDir, maybeMediaDir) of
    (Nothing, _) -> printf "fatal: new '.ki' directory not empty"
    (_, Nothing) -> printf "fatal: new '_media' directory not empty"
    -- Write repository contents and commit.
    (Just kiDir, Just mediaDir) -> writeInitialCommit colFile targetDir kiDir mediaDir colFileMD5
  where
    gitIgnore = absify targetDir </> Path.Internal.Path ".gitignore" :: Path Abs File


writeInitialCommit :: Path Extant File -> Path Extant Dir -> Path Extant Dir -> Path Extant Dir -> MD5Digest -> IO ()
writeInitialCommit colFile targetDir kiDir mediaDir colFileMD5 = do
  windowsLinks <- writeRepo colFile targetDir kiDir mediaDir
  repo <- openLgRepository $ defaultRepositoryOptions { repoPath = toFilePath targetDir }
  return ()


main :: IO ()
main = print "Hello"
