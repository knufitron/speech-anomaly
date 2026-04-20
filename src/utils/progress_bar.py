def progress_bar(current: int, total: int, bar_length: int = 28) -> None:
	total = max(total, 1)
	fraction = current / total
	arrow = int(fraction * bar_length) * "#"
	padding = (bar_length - len(arrow)) * " "
	end = "" if fraction < 1 else "\n"
	print(f"\rProgress: [{arrow}{padding}] {int(fraction * 100):3d}%", end=end, flush=True)
