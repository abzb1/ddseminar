nvidia-smi

for j in $(seq 0 3)
do
    for i in $( lsof /dev/nvidia$j | grep python | awk '{print $2}' | sort -u); do kill -9 $i; done
done

echo "kill pretrain zombie"

for i in $( ps -ef | grep python | grep bert | awk '{print $2}'); do kill -9 $i; done

for i in $( ps -ef | grep python | grep bert | awk '{print $2}'); do kill -9 $i; done


nvidia-smi
