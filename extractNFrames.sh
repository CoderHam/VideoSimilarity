#!/bin/sh
rm -rf data/tmp/*.jpg
total_frames=`ffprobe -loglevel panic -show_streams $1 -select_streams v | grep "^nb_frames" | cut -d '=' -f 2`
# echo $total_frames
numframes=$2
rate=`echo "scale=0; $total_frames/$numframes" | bc`
# echo $rate
ffmpeg -loglevel panic -i $1 -f image2 -vf scale="426x240","select='not(mod(n,$rate))'" -vframes $numframes -vsync vfr data/tmp/output_%03d.jpg
