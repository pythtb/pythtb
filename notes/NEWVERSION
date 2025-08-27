1. I will assume that up to this point the candidate for new
release is in develop branch on the git.  Also, I will assume that we have
tested already the website to some extent and that the code can run all the
examples when creating the website, and that all tox tests are passing.

2. Now let us create a release branch off of develop:

  git checkout develop
  git checkout -b release/1.x.x

3. Any changes we now make to this release candidate should go to the
release/1.x.x. branch.

4a. Update __version__ variable and date in the header of pythtb.py.
Also update the year in the line below starting with "Copyright"

  git add pythtb.py

4b. Update version string in setup.py

  git add setup.py

4c. Update website/local/release/release.rst by specifying what is new in the
package.

  git add website/local/release/release.rst

4d. Add folder in website/local/release/ver_ABC for the old version of the package,
put in .tar.gz file. of the old version.
*It is enough to do this for the old version only!*
New version will be added automatically by the "go" script in website folder.
To get old source file go to this website:
https://pypi.org/project/pythtb/1.OLD.OLD/#files
and download the source.  Then do this,

  git add ver_1.OLD.OLD
  git add ver_1.OLD.OLD/pythtb-1.OLD.OLD.tar.gz

4e. Update website/source/install.rst with the new version. It is enough to
change string "1.6.1" with "1.6.2" or similar.

  git add install.rst

4f. Update website/source/conf.py. Version number appears at two places, just
update the string.

4g. There is NO need to update src/CHANGES. We may want to remove that
file in the future.

4h. Do a quick grep on the old version number just to make sure there
isn't something new that needs to be done.

5. We should test the website to make sure it is rendered correctly. David
should do this test on release/1.x.x branch

   git checkout release/1.x.x

6. Sinisa will wait for David's confirmation that website is running well
on his end before proceeding to the next step.  David can push to release/1.x.x
if he has any additional corrections.

7. When we are happy with polishing release/1.x.x, we should merge
it into develop and master (checking for and resolving conflicts if develop
has been changed).

8. Now Sinisa will update the code on the PyPI server.

8a. Make sure that you link ../../private/.pypirc_MOVE_TO_HOME to home folder as ~/.pypirc  !
This file contains a passwords so I'm not keeping it on github.

8b. Here are instructions for testing purposes only:

  git checkout master
  git pull
  rm -rf dist
  python setup.py sdist
  twine upload dist/* -r testpypi

8c. If you wish to test the package do this,

  git checkout master
  git pull
  pip install -i https://test.pypi.org/simple/ pythtb==1.8.0

8d. When you are sure that this works you can officially upload it to pypi like this.
Note that this code below should not be executed lightly! These lines will make
some changes in the pypi servers and after that it is hard to tweak things.
Therefore make sure you do all the tests you need to do first with the 
"testpypi" version!  Once you are happy with how things look like execute
lines below:

******************* BE CAREFUL WITH THIS *************************

  git checkout master
  git pull
  rm -rf dist
  python setup.py sdist
  twine upload dist/* -r pypi
  
******************************************************************

9. David should now do the final update of the website from the "master" branch of git.  David,
please add here more information if needed.

   git checkout master
   git pull
   ...

10. Sinisa will now make sure that conda-forge is up to date.
Please follow instructions here: https://conda-forge.org/docs/maintainer/adding_pkgs/
Once you do, please list here what had to be done to update conda-forge.

11. Update this file if needed.