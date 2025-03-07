dataset_path="./datasets/Ithaca365/Ithaca365-sub"
save_temp_path="./datasets/Ithaca365/Ithaca365-256"
save_path="./datasets/Ithaca365/Ithaca365-scenario"
scene_json_path="./datasets/Ithaca365/v2.2/scene.json"
weather_json_path="./datasets/Ithaca365/v2.2/weather.json"

python ./scripts/resize.py --dataset_path $dataset_path --save_path $save_temp_path
echo "Finish resize!"
python ./scripts/ithaca_split2scenario.py --dataset_path $save_temp_path --save_path $save_path --scene_json_path $scene_json_path --weather_json_path $weather_json_path
echo "Finish split!"
