#/bin/sh
total_frames=`ffprobe -loglevel panic -show_streams $1 | grep "^nb_frames" | cut -d '=' -f 2`
numframes=$2
rate=`echo "scale=0; $total_frames/$numframes" | bc`
# echo $rate
ffmpeg -loglevel panic -i $1 -f image2 -vf scale="426x240","select='not(mod(n,$rate))'" -vframes $numframes -vsync vfr data/tmp/output_%03d.jpg
