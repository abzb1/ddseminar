# GROVER 모델의 DDP 수행

## 사전 준비사항

1. `megatron-latest.sqsh` 컨테이너 재다운로드    
   `megatron-latest.sqsh`: GROVER 모델을 학습할 때 필요한 추가 패키지가 설치된 컨테이너    
   해당 컨테이너를 `/scratch/enroot/megatron-latest.sqsh` 경로에 두었음(이전 컨테이너와 동일한 경로, 파일이름)
   
    ```
    cp /scratch/enroot/megatron-latest.sqsh ~/ddseminar/image
    ```
2. `grover_large.pt`: 사전학습된 GROVER 모델을 다운로드
   
   ```
   cp /scratch/enroot/grover_large.pt ~/ddseminar/grover_ddp
   ``` 
3. `grover_ddp` 폴더로 들어가 로그를 담을 디렉토리 생성(`_log`, `_err`, `_out`)
   ```
   cd ~/ddseminar/grover_ddp
   mkdir ~/ddseminar/grover_ddp/_log
   mkdir ~/ddseminar/grover_ddp/_err
   mkdir ~/ddseminar/grover_ddp/_out
   ```

## 단일 GPU 사용하기
1. `sbatch.sh` 수정
    ```
    ...
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:a10:1
    ...
    GRES="gpu:a10:1"
    ```
2. `conf.sh` 수정
   ```
   ...
   NPROC_PER_NODE=1
   NNODES=1
   ```

3. sbatch.sh 파일 수행하기
   `sbatch sbatch.sh`


## 데이터 병렬적 딥러닝 수행 확인하기
1. 배치 개수 확인하기    
   - Tox21 dataset의 훈련 데이터 개수는 총 6,264개.    
   - NB(단일 GPU 학습 시 배치 개수) = 6,264 / batch_size(기본값 32)    
       - batch_size=32일 때, NB는 196    

   - NB 확인 경로    
     `~/ddseminar/grover_ddp/_log/{SLURM_JOB_ID}/n0xx.out` 파일을 보면
     ```
     ...
     data_length: xx
     ...
     ```
     다중 GPU에서 데이터 병렬적 딥러닝 학습을 하면 xx 값이 GPU 개수만큼 줄어드는 것을 확인할 수 있음

2. GPU 사용 확인하기
   `~/ddseminar/grover_ddp/{SLURM_JOB_ID}/n0xx.gpu` 파일을 보면 GPU 메모리 사용량을 확인할 수 있음    
   아래와 같은 로그가 보인다면 데이터 병렬적 딥러닝이 수행되고 있는 것임     
   노드 내 모든 GPU가 약 4GB의 메모리를 차지하여 학습 진행 중    
   ```
   n084.hpc             Tue Aug 22 11:18:08 2023  470.82.01
   [0] NVIDIA A10       | 47'C,  88 % |  4500 / 22731 MB | tg9812(4498M)
   [1] NVIDIA A10       | 48'C,  90 % |  4174 / 22731 MB | tg9812(4172M)
   [2] NVIDIA A10       | 47'C,  83 % |  4244 / 22731 MB | tg9812(4242M)
   [3] NVIDIA A10       | 47'C,  86 % |  4102 / 22731 MB | tg9812(4100M)
   ```
   
     
   
