
# coding: utf-8

# In[94]:

import SaikyoABC as sk
import math
from PIL import Image, ImageDraw, ImageFont


# In[178]:

class Compare_answers(object):
    u"""SaikyoABCで作られたペンストロークのブロックをその特徴量を用いて答案間で比較し、クラスタリングするためのクラスです。
    出力は
    self.compared_ans:
    [[[変数名1,blocknum],[変数名2,blocknum],....],[[],[],...],....]
    self.compared_ans[cmp_block_num]で、特徴量の比較からans_block_num番目のクラスタに含められた各、ストロークのブロックの
    もともとsaikyoABC(filename)で割り当てられた変数名とその中におけるそのブロックの番号block_numのリストの群を取り出せます。
    
    SaikyoABC(filename)で以降,主に用いるであろうものとしては
    - filename : 読み込んだファイルの名前を返します。
    - blocks[block_num] : 各ブロックに含まれるストロークの番号をリストに格納した配列です。
    - block_range[block_num][Xmin_or_max][Ymin_or_max] : 各ブロックの領域の配列です。
    - block_center[block_num][XorY] : ブロックの中心の座標が得られます。
    - block_average_length[block_num] : ブロック内のストロークの平均の長さを得られます。
    - stroke_or_string : そのブロックが補助線なら０計算式なら１になっています。
    
    
    Compare_answersのインスタンスの変数とメソッドは
    - cmp_ans : [[[変数名1,blocknum],[変数名2,blocknum],....],[[],[],...],....]の形になっています。
                self.compared_ans[cmp_block_num]
                で、cmp_block_num番目のクラスタに含められた
                各ストロークのブロックの,もともとsaikyoABC(filename)で割り当てられた変数名と
                その中におけるそのブロックの番号(block_num)のリスト
                の群を取り出せます。
    - features : ans_block_numに対応する番号の各特徴量のリストです。
                 features[cmp_block_num][fea_num]
                 で、ans_block_numに分類されているブロック群の各特徴量の内、fea_num番目に割り当てられたものを得られます。
                 この特徴量と比較することで、compared_ansを生成します。この時にfeaturesも更新されます。        
    - check_new_ans(SaikyoABCのインスタンスの名前) : 新たな答案データの特徴量を比較し、compared_ansとfeaturesを更新します。
    """
    
    def __init__(self,ans):
        ans.stroke_classifier()
        #print('filename:{0} is readed'.format(ans.filename))
        self.cmp_ans = [[[ans,block_num]]
                             for block_num in range(len(ans.blocks))]
        self.features = [[ans.block_center[block_num]]
                                  for block_num in range(len(ans.blocks))]
        block_of_ans_num = range(len(ans.blocks))
        imagename = ans.filename.replace('csv','gif')
        imagename = imagename.replace('strokes','image')
        image = Image.open(imagename)
        overlayimage = ans.overlay_image(image)
        overlayimage2 = overlay_cmp_ans_image(ans,overlayimage,block_of_ans_num)
        overlayimage2.show()
        file_num = ans.filename.replace('.csv','')
        file_num = file_num.replace('test/strokes','')
        overlayimage2.save('edited_img' + file_num +'.jpg')
        
    
    def check_new_ans(self,ans):
        u"""新しい回答データを加えて現在のブロックと比較します。
        ans = SaikyoABC(filename)
        の形のものです。
        """
        X = 0
        Y = 1
        ans.stroke_classifier()
        block_of_ans_num = [0] * len(ans.blocks)
        u"""各ブロックがcheck_new_ans実行後のcmp_ansのどの要素に分類されるかを記録します。
        イメージを表示するためにのみ用います。"""
        print('filename:{0} is readed'.format(ans.filename))
        
        added_block = []
        u"""既に対応づけが終わったblock_num"""
        added_cmp_ans = []
        u"""既に対応づけが終わったcmp_ans_num"""
        added_block_num = 0
        added_cmp_ans_num = 0
        while(len(added_block) != len(ans.blocks) and len(added_cmp_ans) != len(self.cmp_ans)):
            distance = 10000
            """self.featuresの各要素に対し、ブロックの特徴量を用いた計算結果の内最小のもの"""
            threshhold = 250
            u"""分類時に用いる閾値"""
            for cmp_ans_num in range(len(self.cmp_ans)):
                if cmp_ans_num not in added_cmp_ans:
                    w1 = 1
                    for block_num in range(len(ans.blocks)):
                        if block_num not in added_block:
                            new_distance = math.sqrt((self.features[cmp_ans_num][0][X] - ans.block_center[block_num][X])**2
                                                     +(self.features[cmp_ans_num][0][Y] - ans.block_center[block_num][Y])**2)
                            u"""何らかのdistanceを返す計算をします。
                            用いる特徴量が決定していないので適宜調整の必要あり
                            """
                            if distance > new_distance :
                                distance = new_distance
                                added_block_num = block_num
                                added_cmp_ans_num = cmp_ans_num
                else:
                    continue
            """
            print('distance-----------------------')
            print(distance)
            print('--------------------------------')
            """
            if distance < threshhold:
                block_of_ans_num[added_block_num] = added_cmp_ans_num
                self.cmp_ans[added_cmp_ans_num].append([ans,added_block_num])
                added_block.append(added_block_num)
                added_cmp_ans.append(added_cmp_ans_num)
                #ここで特徴量の更新も必要
            else:
                print(distance)
                break
                
            
        #どのクラスタにも含めるべきでないと判断されたストロークのブロックを末尾に加える処理を行います。
        for block_num in range(len(ans.blocks)):
            if block_num not in added_block:
                block_of_ans_num[block_num] = len(self.cmp_ans)
                self.cmp_ans.append([[ans,block_num]])
                self.features.append([ans.block_center[block_num]]) 
                
 
                
                
        #=============比較後の答案を表示==================================
        u"""確認用で研究には関係ありません。
        edited'No'.jpgという名前のファイルを作成します。"""
        imagename = ans.filename.replace('csv','gif')
        imagename = imagename.replace('strokes','image')
        image = Image.open(imagename)
        overlayimage = ans.overlay_image(image)
        overlayimage2 = overlay_cmp_ans_image(ans,overlayimage,block_of_ans_num)
        overlayimage2.show()
        file_num = ans.filename.replace('.csv','')
        file_num = file_num.replace('test/strokes','')
        overlayimage2.save('edited_img' + file_num +'.jpg')
        #==============================================================
    
    
    
    
#======================比較した答案を可視化するための関数群========================================
    
def draw_cmp_ans_image(ans,block_of_ans_num,image_size=None):
    u"""ブロックのまとまりを画像として出力する。
    -----
    input
       - image_size: 画像サイズ。(幅,高さ)で指定してください。※numpyの行列とは順番が逆なので注意！！
    """
    if image_size is None:
          image_size = np.max(np.array([np.max(s,axis=1) for s in ans.stroke.STROKE]),axis=0)
    image = Image.new("RGBA",image_size,color=(255,255,255,0))
    __draw_cmp_ans_image(ans,image,block_of_ans_num)
    return image
    
    
def overlay_cmp_ans_image(ans,originalimage,block_of_ans_num):
    u"""元の画像にオーバーレイする形で描画する"""
        
    background = originalimage.copy()
    background = background.convert("RGBA")
    foreground = draw_cmp_ans_image(ans,block_of_ans_num,background.size)
    background.paste(foreground,(0,0),foreground)
    return background
    

        
def __draw_cmp_ans_image(ans,image,block_of_ans_num):
    u"""
    ブロックのスケールを画像にプロットします
    """
    draw = ImageDraw.Draw(image)
    X = 0
    Y = 1
    xmin = 0
    xmax = 1
    ymin = 2
    ymax = 3
    font = ImageFont.truetype("arial.ttf", 32)
    #_font = ImageFont.truetype(size = 15)
    for block_num in range(len(ans.block_range)):
        X_xmin = ans.block_range[block_num][X][xmin]
        Y_ymax = ans.block_range[block_num][Y][ymax]
        X_xmax = ans.block_range[block_num][X][xmax]
        Y_ymin = ans.block_range[block_num][Y][ymin]
        c = block_of_ans_num[block_num]
        if c <= 17:
            draw.rectangle(((X_xmin,Y_ymax),(X_xmax,Y_ymin)),fill=(15*c,255,255-15*c,10),outline=(0,0,0))
        elif c >= 18 and c <= 34:
            c = c - 17
            draw.rectangle(((X_xmin,Y_ymax),(X_xmax,Y_ymin)),fill=(0,15*c,255-15*c,10),outline=(0,0,0))
        else:
            draw.rectangle(((X_xmin,Y_ymax),(X_xmax,Y_ymin)),fill=(255,255,255,10),outline=(0,0,0))
        draw.text((X_xmin,Y_ymax), str(block_of_ans_num[block_num]) ,fill='#000',font = font)



            

        
        


# In[179]:

abc = sk.SaikyoABC("test/strokes8.csv")
ABC = Compare_answers(abc)
#print(ABC.cmp_ans)


# In[180]:

d = sk.SaikyoABC("test/strokes6.csv")
ABC.check_new_ans(d)
#print(ABC.cmp_ans)


# In[181]:

e = sk.SaikyoABC("test/strokes3.csv")
ABC.check_new_ans(e)
#print(ABC.cmp_ans[2])


# In[182]:

f = sk.SaikyoABC("test/strokes10.csv")
ABC.check_new_ans(f)


# In[183]:

g = sk.SaikyoABC("test/strokes11.csv")
ABC.check_new_ans(g)


# In[184]:

h = sk.SaikyoABC("test/strokes12.csv")
ABC.check_new_ans(h)


# In[185]:

i = sk.SaikyoABC("test/strokes14.csv")
ABC.check_new_ans(i)


# In[186]:

j = sk.SaikyoABC("test/strokes7.csv")
ABC.check_new_ans(j)


# In[187]:

print(ABC.cmp_ans)


# In[192]:

print(ABC.cmp_ans[0][2][0].filename)


# In[ ]:




# In[ ]:



