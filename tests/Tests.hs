{-# LANGUAGE OverloadedStrings #-}
import Test.LeanCheck
import Test.LeanCheck.Instances ()
import Test.Hspec (describe, hspec, it, shouldBe)

import qualified Path.Internal

import Ki (maxFilenameSize, getDir)

main :: IO ()
main = hspec $ do
  describe "getDir" $ do
    it "Does something reasonable when given an empty string" $ do
      getDir "" `shouldBe` Path.Internal.Path ""
