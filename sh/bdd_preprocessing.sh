train_path="./datasets/bdd100k/images/100k/train"
save_path="./datasets/bdd100k/weather"
label_json_path="./datasets/bdd100k/images/bdd100k/labels/bdd100k_labels_images_train.json"

python ./scripts/bdd_split2scenario.py --train_path $train_path --save_path $save_path --label_json_path $label_json_path
echo "Finish split!"
