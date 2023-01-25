query1=5
query2=10

# adv_train, proxy
python src/HopSkipJumpAttack.py --AK=robust_white --query=${query1}  --output_path=output/HopSkipJumpAttack/robust_white > output/HopSkipJumpAttack/robust_white/query${query1}.txt
python src/HopSkipJumpAttack.py --AK=robust_white --query=${query2}  --output_path=output/HopSkipJumpAttack/robust_white > output/HopSkipJumpAttack/robust_white/query${query2}.txt

