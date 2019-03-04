# tar extract files first
# iterate for each file and convert to mp4
for folder in ../data/UCF-101/*; do
  for filename in $folder/*; do
    ffmpeg -loglevel panic -i ../data/UCF-101/`basename $folder`/`basename "$filename"` -c:v libvpx-vp9 -threads 4 ../data/UCF101/`basename "${filename%.*}"`.webm
  done
done
