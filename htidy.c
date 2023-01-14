#include <tidy.h>
#include <tidybuffio.h>
#include <stdio.h>
#include <errno.h>


char* tidy(char* input) {
  char *cleansed_buffer_; 

  // uses Libtidy to convert the buffer to XML
  TidyBuffer output = {0};
  TidyBuffer errbuf = {0};
  int rc = -1;
  Bool ok;

  TidyDoc tdoc = tidyCreate();                     // Initialize "document"

  // ok = tidyOptSetBool( tdoc, TidyQuiet, yes ) && ok;
  ok = tidyOptSetValue( tdoc, TidyIndentContent, "true" );
  ok = tidyOptSetBool( tdoc, TidyOmitOptionalTags, yes ) && ok;
  ok = tidyOptSetBool( tdoc, TidyMark, no ) && ok;
  ok = tidyOptSetValue( tdoc, TidyBodyOnly, "yes" ) && ok;
  ok = tidyOptSetInt( tdoc, TidyWrapLen, 120 ) && ok;
  if ( ok )
      rc = tidySetErrorBuffer( tdoc, &errbuf );      // Capture diagnostics
  if ( rc >= 0 )
      rc = tidyParseString( tdoc, input );           // Parse the input
  if ( rc >= 0 )
      rc = tidyCleanAndRepair( tdoc );               // Tidy it up!
  if ( rc >= 0 )
      rc = tidyRunDiagnostics( tdoc );               // Kvetch
  if ( rc > 1 )                                    // If error, force output.
      rc = ( tidyOptSetBool(tdoc, TidyForceOutput, yes) ? rc : -1 );
  if ( rc >= 0 ){
      rc = tidySaveBuffer( tdoc, &output );          // Pretty Print

      // get some mem
      uint yy = output.size;
      cleansed_buffer_ = (char *)malloc(yy+10);
      uint xx = yy+10;
      rc = tidySaveString(tdoc, cleansed_buffer_, &xx );
      if (rc == -ENOMEM)
          printf("fatal: tidy returned -ENOMEM\n");

  }
  tidyBufFree( &output );
  tidyBufFree( &errbuf );
  tidyRelease( tdoc );
  return cleansed_buffer_;
}
