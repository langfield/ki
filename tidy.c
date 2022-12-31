#include <tidy.h>
#include <tidybuffio.h>
#include <stdio.h>
#include <errno.h>

char* tidy(char* input)
{
  TidyBuffer output = {0};
  TidyBuffer errbuf = {0};
  int rc = -1;
  Bool ok;

  TidyDoc tdoc = tidyCreate();                     // Initialize "document"
  printf( "Tidying:\t%s\n", input );

  ok = tidyOptSetBool( tdoc, TidyXhtmlOut, yes );  // Convert to XHTML
  ok = tidyOptSetBool( tdoc, TidyQuiet, yes ) && ok;
  ok = tidyOptSetBool( tdoc, TidyIndentContent, yes ) && ok;
  ok = tidyOptSetBool( tdoc, TidyOmitOptionalTags, yes ) && ok;
  ok = tidyOptSetBool( tdoc, TidyBodyOnly, yes ) && ok;
  ok = tidyOptSetBool( tdoc, TidyWrapLen, 68 ) && ok;
  ok = tidyOptSetBool( tdoc, TidyWrapAttVals, yes ) && ok;
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
  if ( rc >= 0 )
    rc = tidySaveBuffer( tdoc, &output );          // Pretty Print

  if ( rc >= 0 )
  {
    if ( rc > 0 )
      printf( "\nDiagnostics:\n\n%s", errbuf.bp );
    printf( "\nAnd here is the result:\n\n%s", output.bp );
  }
  else
    printf( "A severe error (%d) occurred.\n", rc );

  tidyBufFree( &errbuf );
  tidyRelease( tdoc );
  return &output;
}
