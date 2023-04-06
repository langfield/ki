{-# LANGUAGE OverloadedStrings #-}
import Path (Abs, Dir, Path, toFilePath)
import Test.LeanCheck ()
import Test.LeanCheck.Instances ()
import Test.Hspec (describe, hspec, it, shouldBe, shouldNotBe)

import qualified Path.Internal

import Ki (getDir)

main :: IO ()
main = hspec $ do
  describe "getDir" $ do
    it "Does something reasonable when given an empty string" $ do
      getDir "" `shouldBe` (Path.Internal.Path "./" :: Path a Dir)
    it "Encodes empty string as './', not an empty string (Is this the behavior we want?)" $ do
      getDir "" `shouldNotBe` (Path.Internal.Path "" :: Path a Dir)
    it "Writes the correct data for absolute paths" $ do
      getDir "a" `shouldBe` (Path.Internal.Path "/a" :: Path Abs Dir)
    it "Gives a correct FilePath for absolute paths" $ do
      toFilePath (getDir "a") `shouldBe` "/a"

