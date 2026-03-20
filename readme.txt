# 1. Install vllm from source with precompiled flag
# 2. Declare vllm path 
    export VLLM_PATH=/root/regify/vllm
# 3. Run TLC model checker
    java -Xmx30g -cp tla2tools-json-test.jar tlc2.TLC -noGenerateSpecTE -seed 10 -dump json shared.json -fp 10 -workers auto -deadlock -config SchedulerSharedBlocks.cfg SchedulerSharedBlocks.tla
    python replayer_shared.py --states shared-states.json --edges shared-edges.json --max-paths 7 --target any 2>&1 | tee replayer_shared.log