#!/bin/bash
wget --no-check-certificate "https://onedrive.live.com/download?cid=66D44078EAF6817B&resid=66D44078EAF6817B%211980&authkey=AK39ZUZ-Sy1Pi6I" -O 7001-8000.tar

wget --no-check-certificate "https://onedrive.live.com/download?cid=66D44078EAF6817B&resid=66D44078EAF6817B%211979&authkey=ANqGPPZke0OIHEE" -O 8001-9000.tar

wget --no-check-certificate "https://onedrive.live.com/download?cid=66D44078EAF6817B&resid=66D44078EAF6817B%211978&authkey=AIDTjiKxdL32ou8" -O 9001-10000.tar

tar -xvf 1-1000.tar
rm -rf 1-1000.tar

tar -xvf 1001-2000.tar
rm -rf 1001-2000.tar

tar -xvf 2001-3000.tar
rm -rf 2001-3000.tar

tar -xvf 3001-4000.tar
rm -rf 3001-4000.tar

tar -xvf 4001-5000.tar
rm -rf 4001-5000.tar

tar -xvf 5001-6000.tar
rm -rf 5001-6000.tar

tar -xvf 6001-7000.tar
rm -rf 6001-7000.tar

tar -xvf 7001-8000.tar
rm -rf 7001-8000.tar

tar -xvf 8001-9000.tar
rm -rf 8001-9000.tar

tar -xvf 9001-10000.tar
rm -rf 9001-10000.tar