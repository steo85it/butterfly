#!/usr/bin/env sh

mkdir -p sphere_meshes/h
mkdir -p sphere_meshes/p
mkdir -p sphere_meshes/hp

move_to_dir () {
	mv L_data.bin $1
	mv L_colind.bin $1
	mv L_rowptr.bin $1
	mv M_data.bin $1
	mv M_colind.bin $1
	mv M_rowptr.bin $1
	mv nodes.bin $1
}

fix_files () {
	./fix_CSR_bin_files.py ${1}/L_data.bin ${1}/L_rowptr.bin ${1}/L_colind.bin
	./fix_CSR_bin_files.py ${1}/M_data.bin ${1}/M_rowptr.bin ${1}/M_colind.bin
	./fix_node_files.py ${1}/nodes.bin
}

# fix h to some coarse level (e.g. ~5k triangles) and take p from 1 up
# to ~20 (or whatever the largest p is that gives ~1M degrees of
# freedom)

echo "generating p-refined meshes..."
for i in $(seq 1 8); do
	echo "./lbo_MFEM --elem 0 --order ${i} --refine 5"
	./lbo_MFEM --elem 0 --order ${i} --refine 5
	mkdir -p sphere_meshes/p/${i}
	move_to_dir sphere_meshes/p/${i}
	fix_files sphere_meshes/p/${i}
done

# fix p=1 and for h such that the mesh has ~5k up to ~1M triangles
# (you can replace the existing sphere meshes)

echo "generating h-refined meshes..."
for i in $(seq 4 8); do
	echo "./lbo_MFEM --elem 0 --order 1 --refine ${i}"
	./lbo_MFEM --elem 0 --order 1 --refine ${i}
	mkdir -p sphere_meshes/h/${i}
	move_to_dir sphere_meshes/h/${i}
	fix_files sphere_meshes/h/${i}
done

# pick moderate parameters like p=6 and h such that there are ~100k
# total DOFs

echo "generating one hp mesh..."
echo "./lbo_MFEM --elem 0 --order 5 --refine 4"
./lbo_MFEM --elem 0 --order 5 --refine 4
mkdir -p sphere_meshes/hp
move_to_dir sphere_meshes/hp
fix_files sphere_meshes/hp
