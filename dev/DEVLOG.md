# Development Log for PythTB
This file contains notes from developers that should be brought to the attention of the maintainers or other developers.
For example, if something was removed or added where the context is unclear in the git history, the change should be written 
here with some explanatory notes, along with your name and date.

## Removed dev/TODO.md: Trey - July 1, 2025

I have removed the 'dev/TODO.md' in the dev folder and moved all of its contents to the Issues tab. This way we can comment and track the ideas initiated by Sinisa from this file. In the future, any ideas for future development should be added to "Discussions", and issues to fix should be added to the "Issues" tab.

## Note: David - Nov 18, 2012

If you encounter an error like
```
  ------------------------------------------------------
  ERROR: Error in "xxx" directive: invalid option block.
  ------------------------------------------------------
```
Then 
- (a) leave a blank line between the directive and text block,
  
and/or 

- (b) make sure no line in text block starts with a colon
by rewrapping the text in some way.

(This seems to be a problem that occurs for my version of Docutils
but not Sinisa's...)
