def read_csv_to_lists(file_path):
    result = []
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        # 跳过标题行（如果有的话）
        next(csv_reader, None)
        for row in csv_reader:
            # 从D列（索引3）开始到最后一列
            row_list = row[3:]
            # 移除空字符串
            row_list = [item for item in row_list if item.strip() != '']
            # 如果行不为空，则添加到结果中
            if row_list:
                result.append(row_list)
    return result

# 使用示例
file_path = 'your_csv_file.csv'  # 替换为你的CSV文件路径
data_lists = read_csv_to_lists(file_path)

# 打印结果
for row in data_lists:
    print(row)