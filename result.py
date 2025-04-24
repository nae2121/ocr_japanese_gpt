import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract id and final_label from full CSV")
    parser.add_argument('--input_csv',  required=True, help='抽出元のCSVファイルパス')
    parser.add_argument('--output_csv', required=True, help='出力する提出用CSVファイルパス')
    parser.add_argument('--encoding',       default='utf-8-sig', help='入力CSVの文字エンコーディング（例: utf-8-sig, shift-jis）')
    parser.add_argument('--output_encoding', default='shift-jis', help='出力CSVの文字エンコーディング')

    args = parser.parse_args()

    # 入力CSV読み込み
    df = pd.read_csv(args.input_csv, encoding=args.encoding)

    # 必要列の抽出
    # 「final_label」列がない場合は「label」列を利用
    if 'final_label' in df.columns:
        df_out = df[['id', 'final_label']].copy()
        df_out.columns = ['id', 'label']
    elif 'label' in df.columns:
        df_out = df[['id', 'label']].copy()
    else:
        raise KeyError("入力CSVに 'final_label' もしくは 'label' 列が見つかりません。")

    # idを数値としてソート（数字以外も許容）
    try:
        df_out['id'] = df_out['id'].astype(int)
    except:
        pass
    df_out = df_out.sort_values('id').reset_index(drop=True)

    # 出力先ディレクトリがなければ作成
    out_dir = os.path.dirname(args.output_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # CSV出力
    df_out.to_csv(args.output_csv, index=False, encoding=args.output_encoding)
    print(f"Extracted {len(df_out)} rows to {args.output_csv} ({args.output_encoding})")

if __name__ == "__main__":
    main()