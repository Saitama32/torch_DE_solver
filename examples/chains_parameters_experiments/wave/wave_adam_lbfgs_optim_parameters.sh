#!/bin/bash

pde=wave
seeds=(345 456 567 234 123)
losses=(mse)
n_neurons=(100 200 400)
history_size=(100 200)
n_layers=4
num_x=257
num_t=101
num_res=10000
# opt='[Adam LBFGS]'
lrs=(0.0001 0.001 0.01)
epochs_Adam=1000
epochs_LBFGS=2050
betas=(5)
devices=(0)
proj=wave_adam_lbfgs_parameters_full_rmse
max_parallel_jobs=5

background_pids=()
current_device=0

interrupted=0  # Flag to indicate if Ctrl+C is pressed

# Function to handle SIGINT (Ctrl+C)
cleanup() {
    echo "Interrupt received, stopping background jobs..."
    interrupted=1  # Set the flag
    for pid in "${background_pids[@]}"; do
        kill $pid 2>/dev/null
    done
}

# Trap SIGINT
trap cleanup SIGINT

for seed in "${seeds[@]}"
do
    for loss in "${losses[@]}"
    do
        for n_neuron in "${n_neurons[@]}"
        do
            for beta in "${betas[@]}"
            do
                for lr in "${lrs[@]}"
                do
                    if [ $interrupted -eq 0 ]; then  # Check if Ctrl+C has been pressed
                        device=${devices[current_device]}
                        current_device=$(( (current_device + 1) % ${#devices[@]} ))

                        python wave_run_experiment.py --seed $seed --pde $pde --pde_params beta $beta --opt Adam LBFGS \
                            --opt_params_Adam lr $lr --opt_params_LBFGS history_size $history_size --num_layers $n_layers --num_neurons $n_neuron \
                            --loss $loss --num_x $num_x --num_t $num_t --num_res $num_res --epochs_Adam $epochs_Adam --epochs_LBFGS $epochs_LBFGS  --comet_project $proj \
                            --device $device &

                        background_pids+=($!)

                        # Limit the number of parallel jobs
                        while [ $(jobs | wc -l) -ge $max_parallel_jobs ]; do
                            wait -n
                            # Clean up finished jobs from the list
                            for i in ${!background_pids[@]}; do
                                if ! kill -0 ${background_pids[$i]} 2> /dev/null; then
                                    unset 'background_pids[$i]'
                                fi
                            done
                        done
                    fi
                done
            done
        done
    done
done

# Wait for all background jobs to complete
wait

# Cleanup on normal exit
cleanup