export MACHINE=0
python pipe.py $MACHINE -s 2048 -b 1
python pipe.py $MACHINE -s 1024 -b 1
python pipe.py $MACHINE -s 512 -b 1
python pipe.py $MACHINE -s 256 -b 1
python pipe.py $MACHINE -s 128 -b 1


python pipe.py $MACHINE -s 2048 -b 8
python pipe.py $MACHINE -s 1024 -b 8
python pipe.py $MACHINE -s 512 -b 8
python pipe.py $MACHINE -s 256 -b 8
python pipe.py $MACHINE -s 128 -b 8
