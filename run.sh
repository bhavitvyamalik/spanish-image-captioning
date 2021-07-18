read -p "Enter repo name: " repo

if [ -d $repo ]
then
    cd $repo
    git pull
    cd ..
else
    git clone https://huggingface.co/flax-community/$repo
fi

token=$(cat /home/bhavitvya_malik/.huggingface/token)

echo "Token found $token"
python3 main.py \
    --output_dir $repo \
    --seed 42 \
    --logging_steps 300 \
    --eval_steps 600 \
    --save_steps 2000 \
    --data_dir /home/user/data/CC12M/images \
    --train_file /home/chhablani_gunjan/spanish-image-captioning/data/train_file_es.tsv \
    --validation_file /home/chhablani_gunjan/spanish-image-captioning/data/val_file_es.tsv \
    --save_total_limit 6 \
    --push_to_hub \
    --num_train_epochs 10 \
    --push_to_hub_organization flax-community \
    --push_to_hub_token $token \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --preprocessing_num_workers 16 \
    --warmup_steps 500 \