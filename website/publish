#!/bin/bash

# publish operation must be done from David's home desktop
if [ `hostname` != "desktop-dell" ];then
  echo "Error: Must be executed from David's home desktop"
  exit
fi

# get version name
version=`grep version source/conf.py | grep = | awk -F\' '{print $2}'`
name=pythtb_website_$version

############################
# optional renaming
name=$name-newdoc
############################

cd build/html
tar -hcvf $name.tar *

mv $name.tar ~/pu/pythtb-master/

printf "$name.tar moved to ~/pu/pythtb-master/\n"
