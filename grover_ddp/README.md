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
   
