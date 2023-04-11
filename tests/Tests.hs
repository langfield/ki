{-# LANGUAGE OverloadedStrings #-}
import Path (Abs, Dir, Path, Rel, toFilePath)
import Test.LeanCheck ()
import Test.LeanCheck.Instances ()
import Test.Hspec (describe, hspec, it, shouldBe, shouldNotBe)

import qualified Path.Internal

import Ki (getDir, castDir)

main :: IO ()
main = hspec $ do
  describe "'getDir :: Text -> Path a Dir'" $ do
    it "does something reasonable when given an empty string" $ do
      getDir "" `shouldBe` (Path.Internal.Path "./" :: Path a Dir)
    it "encodes empty string as './', not an empty string (Is this the behavior we want?)" $ do
      getDir "" `shouldNotBe` (Path.Internal.Path "" :: Path a Dir)
    it "writes the correct data for absolute paths" $ do
      getDir "a" `shouldBe` (Path.Internal.Path "/a" :: Path Abs Dir)
    it "gives a correct FilePath for absolute paths" $ do
      toFilePath (getDir "a" :: Path Abs Dir) `shouldBe` "/a"
    it "strips pathseps" $ do
      getDir "a/b" `shouldBe` (Path.Internal.Path "ab" :: Path Rel Dir)
  describe "'castDir :: Path b Dir -> Path b' Dir'" $ do
    it "casts an absolute to a relative" $ do
      castDir (Path.Internal.Path "a" :: Path Abs Dir) `shouldBe` (Path.Internal.Path "a" :: Path Rel Dir)
    it "casts a relative to an absolute" $ do
      castDir (Path.Internal.Path "a" :: Path Rel Dir) `shouldBe` (Path.Internal.Path "a" :: Path Abs Dir)
