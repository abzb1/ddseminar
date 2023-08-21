# GROVER 모델의 DDP 수행

`grover_ddp` 폴더로 들어가 로그를 담을 디렉토리 생성(`_log`, `_err`, `_out`)
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
   
