(. ./setups.sh

python3 compare.py "${setups[@]}" "${completed_setups[@]}" 
echo "compare.sh executed"

)