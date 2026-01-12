bash build.sh && \
ssh -i ~/.ssh/mry_id_ed25519 root@10.129.131.45 "rm -rf /root/mry/cxl_shm_test/cxl_shm_p2p_test && mkdir -p /root/mry/cxl_shm_test/cxl_shm_p2p_test" && \
scp -i ~/.ssh/mry_id_ed25519 -r /root/mry/cxl_shm_test/cxl_shm_p2p_test root@10.129.131.45:/root/mry/cxl_shm_test && \
ssh -i ~/.ssh/mry_id_ed25519 root@10.129.131.45 "cd /root/mry/cxl_shm_test/cxl_shm_p2p_test && bash build.sh" && \
bash run_active.sh 
# sleep 3 && \
# ssh -i ~/.ssh/mry_id_ed25519 root@10.129.131.45 "cd /root/mry/cxl_shm_test/cxl_shm_p2p_test && bash run_passive.sh" 