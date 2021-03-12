#/bin/bash
#
# If you get permission error, you can try
# chmod +rx autoformat.sh 

echo 'Running isort'
isort -rc ./src

echo 'Running black'
black ./src

echo 'Finished auto formatting'