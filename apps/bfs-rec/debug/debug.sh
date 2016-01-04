#./gpu-bfs-rec -f 0 -i ../old/test.gr -s 0
#./gpu-bfs-rec -f 0 -i ../old/test.gr -s 1
#./gpu-bfs-rec -f 0 -i ../old/test.gr -s 2
#./gpu-bfs-rec -f 0 -i ../old/test.gr -s 3

#./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/p2p-Gnutella08.txt -e 1 -v -s 2
#./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/p2p-Gnutella31.txt -v -s 0
#./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/p2p-Gnutella31.txt -v -s 1
#./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/p2p-Gnutella31.txt -v -s 2
#cuda-memcheck ./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/p2p-Gnutella31.txt -v -s 3
#./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/p2p-Gnutella31.txt -v -s 3

#./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/wiki-Vote.txt -v -s 0
#./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/wiki-Vote.txt -v -s 1
#./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/wiki-Vote.txt -v -s 2
#cuda-memcheck ./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/p2p-Gnutella31.txt -v -s 3
#./gpu-bfs-rec -f 2 -i ../../../../datasets/SLNDC/wiki-Vote.txt -v -s 3


#./gpu-bfs-rec -f 1 -i ../../../../datasets/DIMACS10/kron_g500-simple-logn16.graph -v -s 0
#./gpu-bfs-rec -f 1 -i ../../../../datasets/DIMACS10/kron_g500-simple-logn16.graph -v -s 1
#./gpu-bfs-rec -f 1 -i ../../../../datasets/DIMACS10/kron_g500-simple-logn16.graph -v -s 2
./gpu-bfs-rec -f 1 -i ../../../../datasets/DIMACS10/kron_g500-simple-logn16.graph -v -s 3
