#!/bin/bash
# Автоматический запуск poisson_2d_cg_chain.py с распределением по GPU

SCRIPT="examples/examples_PINNacle/example_poisson_PINNacle/poisson_2d_ms_pinncale_comparison.py"

# Проверяем, сколько доступно GPU
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

log_enable="True"
log_unenable="False"
exp_key_1="bf05c571f59540a195ffc82e165f06ac"
exp_key_2="fefe058185f547af81e787cf603bd4eb"
echo "Обнаружено GPU: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "❌ Не найдено ни одного CUDA-устройства. Выходим."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Запускаем 2 процесса на одной GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_unenable" --exp_key "$exp_key_1"&
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_unenable" --exp_key "$exp_key_2"&
elif [ "$NUM_GPUS" -ge 2 ]; then
    echo "Запускаем по 2 процесса на каждую из двух GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_unenable" --exp_key "$exp_key_1"&
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$log_unenable" --exp_key "$exp_key_2"&
    # CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$log_enable" --exp_key "$exp_key_3"&
    # CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$log_enable" --exp_key "$exp_key_4"&
else
    echo "⚠️ Найдено более 2 GPU, но используется только первые две."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" &
fi

# Ждём завершения всех процессов
wait
echo "✅ Все процессы завершены."
