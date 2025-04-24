export WANDB_API_KEY=cbacbd4d69837b2f36b650e8682607b8910fa76c
#matched_paths=($1)
#echo $1
#
## 遍历所有匹配的路径
#for path in "${matched_paths[@]}"; do
#    echo "匹配的路径: $path"
#done
wandb sync --entity 724183484zz-personal $1
