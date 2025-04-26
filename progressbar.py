from tqdm import tqdm as bar
class ProgressBar:
    def __init__(self, total:int=100, desc:str="Loading...", unit:str="s", color:int=0, length:int = 50, other:str=None) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.color = self.get_color(color)
        self.bar = bar(total=total, desc=desc, unit=unit, colour=self.color, dynamic_ncols=False, ncols=65+length, ascii=" ░▒▓█")
        self.other = other   
        return
    def update(self, increase=1, other:float=None) -> None:
        if other is not None and self.other is not None:
            self.bar.set_description(f"{self.desc} | {self.other}: {other:.1f}")

        self.bar.update(increase)
        if self.bar.n >= self.total:
            self.bar.close()
        return
    def get_color(self, index: int=0) -> str:
        colors = ['WHITE','BLACK', 'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN']
        if index >= len(colors) or index < 0:
            print(f"Choose index from colors[0-{len(colors)-1}]: ", colors)
            index = 0
        return colors[index]