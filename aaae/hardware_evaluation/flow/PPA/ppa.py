import re

def extract_timing(filename="timing.txt"):
    """提取 data arrival time 的数值"""
    with open(filename, 'r') as f:
        for line in f:
            m = re.search(r'([\d\.Ee\+\-]+)\s+data arrival time', line)
            if m:
                return float(m.group(1))
    return None


def extract_area(filename="area.txt"):
    """提取芯片面积数值"""
    with open(filename, 'r') as f:
        for line in f:
            m = re.search(r"Chip area for top module.*:\s*([\d\.Ee\+\-]+)", line)
            if m:
                return float(m.group(1))
    return None


def extract_power(filename="power.txt"):
    """提取 Combinational 的 Total Power 数值"""
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith("Combinational"):
                parts = line.split()
                if len(parts) >= 5:
                    return float(parts[4])
    return None


if __name__ == "__main__":
    timing = extract_timing()
    area = extract_area()
    power = extract_power()

    if timing is None or area is None or power is None:
        print("⚠️ Warning: 部分值未提取成功，请检查输入文件格式。")
    else:
        print(f"提取成功: timing={timing}, area={area}, power={power}")

    # === 写入 ppa.log (追加模式) ===
    with open("ppa.log", "a") as f:
        f.write(f"{timing:.6f} {area:.6f} {power:.6e}\n")
