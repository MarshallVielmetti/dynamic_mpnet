#!/bin/bash

# Default values
NUM_INSTANCES=1
SCRIPT_NAME="code/rrt_data_gen.py"

# Function to display usage information
function display_usage {
  echo "Usage: $0 -n <num_instances> [-- python_script_args]"
  echo "  -n: Number of instances to run (default: 1)"
  echo "  Use -- to separate script options from Python args (optional)"
  echo ""
  echo "Example: $0 -n 4 -- --num_maps 10 --num_samples_per 5"
  exit 1
}

# Handle arguments
PYTHON_ARGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n)
      if [[ -z "$2" || "$2" =~ ^- ]]; then
        echo "Error: -n requires a numeric argument"
        display_usage
      fi
      NUM_INSTANCES="$2"
      shift 2
      ;;
    -h|--help)
      display_usage
      ;;
    --)
      shift
      PYTHON_ARGS="$*"
      break
      ;;
    *)
      PYTHON_ARGS="$*"
      break
      ;;
  esac
done

# Ensure NUM_INSTANCES is a valid number
if ! [[ "$NUM_INSTANCES" =~ ^[0-9]+$ ]]; then
  echo "Error: Number of instances must be a positive number"
  exit 1
fi

# Print execution summary
echo "Starting $NUM_INSTANCES Docker containers"
echo "Python script: $SCRIPT_NAME"
echo "Python arguments: $PYTHON_ARGS"
echo ""

# Container tracking arrays
container_ids=()
start_times=()

# Start all containers
echo "Launching containers..."
for ((i=1; i<=$NUM_INSTANCES; i++)); do
  echo -n "  Container $i/$NUM_INSTANCES: "
  start_time=$(date +%s)
  container_id=$(docker run -d -v "$(pwd):/app" dynamic_mpnet $SCRIPT_NAME $PYTHON_ARGS)
  
  if [ $? -eq 0 ]; then
    echo "Started (ID: ${container_id:0:12})"
    container_ids+=($container_id)
    start_times+=($start_time)
  else
    echo "Failed to start container $i"
    exit 1
  fi
done

echo -e "\nAll containers started. Monitoring progress..."

# Function to draw progress bar
function draw_progress_bar {
  local width=50
  local percent=$1
  local completed_length=$((width * percent / 100))
  local remaining_length=$((width - completed_length))
  
  printf "["
  printf "%0.s#" $(seq 1 $completed_length)
  printf "%0.s-" $(seq 1 $remaining_length)
  printf "] %d%% " $percent
}

# Monitor containers until all complete
completed=0
while [ $completed -lt $NUM_INSTANCES ]; do
  completed=0
  running=0
  failed=0
  
  for i in "${!container_ids[@]}"; do
    container_id=${container_ids[$i]}
    status=$(docker inspect --format='{{.State.Status}}' $container_id 2>/dev/null)
    
    if [[ "$status" == "exited" ]]; then
      exit_code=$(docker inspect --format='{{.State.ExitCode}}' $container_id 2>/dev/null)
      if [[ "$exit_code" == "0" ]]; then
        ((completed++))
      else
        ((failed++))
      fi
    elif [[ "$status" == "running" ]]; then
      ((running++))
    fi
  done
  
  # Update progress bar
  total_processed=$((completed + failed))
  percent=$((total_processed * 100 / NUM_INSTANCES))
  
  current_time=$(date +%s)
  elapsed=$((current_time - start_times[0]))
  
  printf "\r"
  draw_progress_bar $percent
  printf "| Complete: %d | Running: %d | Failed: %d | Time: %02d:%02d " $completed $running $failed $((elapsed/60)) $((elapsed%60))
  
  # Exit loop if all containers have completed or failed
  if [ $total_processed -ge $NUM_INSTANCES ]; then
    break
  fi
  
  sleep 1
done

echo -e "\n\nExecution summary:"
echo "  Total containers: $NUM_INSTANCES"
echo "  Completed successfully: $completed"
echo "  Failed: $failed"
echo "  Total runtime: $(date -u -d @${elapsed} +"%T")"

# Optional: Show logs from completed containers
if [ $completed -gt 0 ] || [ $failed -gt 0 ]; then
  echo ""
  read -p "Do you want to display logs from containers? (y/n): " show_logs
  if [[ "$show_logs" == "y" || "$show_logs" == "Y" ]]; then
    for container_id in "${container_ids[@]}"; do
      exit_code=$(docker inspect --format='{{.State.ExitCode}}' $container_id 2>/dev/null)
      echo -e "\n===== Container ${container_id:0:12} (Exit: $exit_code) ====="
      docker logs $container_id
    done
  fi
fi

# Clean up containers
echo ""
read -p "Do you want to remove all containers? (y/n): " remove_containers
if [[ "$remove_containers" == "y" || "$remove_containers" == "Y" ]]; then
  docker rm ${container_ids[@]} >/dev/null
  echo "All containers removed."
fi
