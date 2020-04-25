
#!/bin/bash

cd tools
resetflag=""

[ -d HTPA32x32d ] && resetflag=true
if [ "$resetflag" == true ]
then
    read -p "HTPA32x32d exists. Continuing means deleting HTPA32x32d and its content. Continue? [y/n] " -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "Deleting HTPA32x32d."
        rm -r HTPA32x32d
    else
        echo "Aborting setup."
        exit 1
    fi
fi

git clone https://github.com/igor-morawski/HTPA32x32d
find ./HTPA32x32d -mindepth 1 ! -regex '^./HTPA32x32d/tools.py\(/.*\)?' -delete
cd ..
echo "Setup completed!"
