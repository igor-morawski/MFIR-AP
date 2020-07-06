#test, report, compare
(. ./setups.sh

for setup in "${setups[@]}"; do 
    python3 test.py $setup
    python3 report.py $setup
done
echo "Subroutine test.py executed"
echo "Subroutine test.py executed"
python3 compare.py "${setups[@]}"
echo "Subroutine compare.py executed"
echo "test.sh executed"
)