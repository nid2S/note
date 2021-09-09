# print
***
- print(a,b,c,sep=',') > »çÀÌ¸¦ sep ·Î
- print(a,end='') > ³¡À» end ·Î
- ¹®ÀÚ¿­Àº \·Î ÁÙ¹Ù²ÞÀ» ÇÑ µÚ ÀÔ·ÂÇØ ¿©·¯ÁÙ¿¡¼­ ÀÔ·ÂÇÒ ¼ö ÀÖ´Ù.

# comment
- \# : ÆÄÀÌ½ãÀÇ ÁÖ¼®. ÄÚµå ½ÇÇà½Ã ÀÎ½ÄÇÏÁö ¾ÊÀ½.
- \# TODO : PyCharmÀÇ ÇÒÀÏ °ü¸® ±â´É. ÆÄÀÌÂüÀÇ TODOÇ×¸ñ¿¡¼­ ÇÑ´«¿¡ º¼ ¼ö ÀÖÀ½. todo°¡ ÀÖÀ¸¸é Ä¿¹ÔºÒ°¡.
- """ """ : docstring. Å¬·¡½º/¸Þ¼­µå/ÇÔ¼öµîÀÇ »ç¿ëÀÚ°¡ ¾Ë¾Æ¾ß ÇÒ ¼³¸íÀ» Ãß°¡. ÇØ´ç°´Ã¼.__doc\_\_·Î È®ÀÎ°¡´É. ''' '''·Îµµ »ç¿ëÀº °¡´É.
- annotation : Å¬·¡½º/¸Þ¼­µå/ÇÔ¼ö ¿¡¼­ ÀÔ·Â°ª°ú ¹ÝÈ¯°ªÀÇ Á¸Àç/ÀÚ·áÇüÀ» ¾Ë·ÁÁÖ´Â ÁÖ¼®. [ÇÔ¼ö(º¯¼ö: ÀÚ·áÇü) -> ¹ÝÈ¯ÀÚ·áÇü: ÄÚµå] À¸·Î »ç¿ë.

#sequence
***
- °³Ã¼ : entity. Å¬·¡½º¿¡ ¼ÓÇÑ °ª(¼Ó¼º), ¸Þ¼­µå µîÀ» ÀÇ¹ÌÇÔ.
- °´Ã¼ : object. ¾î¶°ÇÑ ¼Ó¼º°ª°ú Çàµ¿À» °¡Áö°í ÀÖ´Â µ¥ÀÌÅÍ(==Å¬·¡½º()). ÆÄÀÌ½ãÀÇ ¸ðµç°Í.
  
- ½ÃÄö½ºÀÚ·áÇü : ÅõÇÃ,¸®½ºÆ®,range,¹®ÀÚ¿­µî °ªÀÌ ¿¬¼ÓÀûÀ¸·Î ÀÌ¾îÁø ÀÚ·áÇü
- ½ÃÄö½º °´Ã¼ : ½ÃÄö½º ÀÚ·áÇüÀ¸·Î ¸¸µç °´Ã¼. ¿ä¼Ò´Â ½ÃÄö½º °´Ã¼¼Ó °¢ °ª. a[0\]½ÄÀ¸·Î ¿ä¼Ò¿¡ Á¢±Ù °¡´É
- ¸®½ºÆ®, Æ©ÇÃ, µñ¼Å³Ê¸®, ¼¼Æ®¸¦ ÄÁÅ×ÀÌ³Ê¶ó°í ºÎ¸§.
  
- id(°´Ã¼) : ÇØ´ç °´Ã¼ÀÇ ÁÖ¼Ò°ª ¹ÝÈ¯.

##list
***
- s=list()/[]
- ra=[15,25,35\]               # ¸®½ºÆ® ÆÐÅ·. ÅõÇÃÀÇ °æ¿ì ÅõÇÃ ÆÐÅ·
- ¸®½ºÆ® = [°ª, °ª, °ª\]         # ¸®½ºÆ® ¸¸µé±â
- ¸®½ºÆ® = []                   # ºó ¸®½ºÆ® ¸¸µé±â
- ¸®½ºÆ® = list()               # ºó ¸®½ºÆ® ¸¸µé±â
- ¸®½ºÆ® = list(range(È½¼ö))     # range ·Î ¸®½ºÆ® ¸¸µé±â

### list attribute
***
- ¸®½ºÆ®.append(¿ä¼Ò) = ¸®½ºÆ® ³¡¿¡ ¿ä¼Ò ÇÏ³ª Ãß°¡. append([])·Î ¸®½ºÆ® ¾È¿¡ ¸®½ºÆ®¸¦ ³ÖÀ» ¼ö ÀÖÀ½. ÀÌ ¸®½ºÆ® ¾È¿¡ ¿ä¼Ò¸¦ ³ÖÀ¸·Á¸é ¸®½ºÆ®[0\].append().
- ¸®½ºÆ®.extend(¸®½ºÆ®) = ¸®½ºÆ®¿¡ ¸®½ºÆ® ¿¬°á,È®Àå.
- ¸®½ºÆ®.insert(ÀÎµ¦½º,¿ä¼Ò) = ÀÎµ¦½º¿¡ ¿ä¼Ò Ãß°¡. ¸®½ºÆ®[a:a\] = [s:d\]·Î Æ¯Á¤ ÀÎµ¦½º¿¡ ¸®½ºÆ® »ðÀÔ °¡´É

- ¸®½ºÆ®.pop() = ¸®½ºÆ®¿¡¼­ ¸¶Áö¸· ¿ä¼Ò »èÁ¦ ÈÄ »èÁ¦µÈ ¿ä¼Ò ¹ÝÈ¯. pop(ÀÎµ¦½º)·Î ÀÎµ¦½ºÀÇ ¿ä¼Ò »èÁ¦. del(¸®½ºÆ®[ÀÎµ¦½º\])·Î ÇØµµ ¹«¹æ.
- ¸®½ºÆ®.remove(°ª) = ¸®½ºÆ®¿¡¼­ °ªÀ» Ã£¾Æ »èÁ¦. Áßº¹ÀÌ ÀÖÀ¸¸é Ã³À½ÀÇ ÇÏ³ª¸¸ »èÁ¦.

- ¸®½ºÆ®.index(°ª) = ¸®½ºÆ®¿¡¼­ °ªÀÇ ÀÎµ¦½º¸¦ Ã£¾ÆÁÜ.
- ¸®½ºÆ®.count(°ª) = ¸®½ºÆ®¿¡¼­ °ªÀÇ °³¼ö¸¦ ±¸ÇÔ.
- ¸®½ºÆ®.reverse() = ¸®½ºÆ® µÚÁý±â

- ¸®½ºÆ®.sort(reverse=False / reverse=True) = ¿À¸§/³»¸²Â÷¼ø Á¤·Ä. ÀÚ±âÀÚ½ÅÀÌ Á¤·ÄµÊ.
- sorted(¸®½ºÆ®) = Á¤·ÄµÈ »õ ¸®½ºÆ® »ý¼º. key=lambda item:¼ö½Ä, reverse=bool µîÀ» ÀÎÀÚ·Î ÁÖ¾î Á¤·Ä ±âÁØ, ¹ÝÀü ¿©ºÎ µîÀ» ¼±ÅÃÇÒ ¼ö ÀÖÀ½.

- ¸®½ºÆ®.clear() = del ¸®½ºÆ®[:\] = ¸®½ºÆ® ¸ðµç ¿ä¼Ò »èÁ¦,
- ¸®½ºÆ®2 = ¸®½ºÆ®1.copy() == ¸®½ºÆ® º¹Á¦. ¸®½ºÆ®2=¸®½ºÆ®1´Â ÇÒ´çÀ¸·Î, ¸®½ºÆ® 1°ú 2°¡ °¡¸®Å°´Â ¸®½ºÆ®´Â µ¿ÀÏÇÔ.
- for b,a in enumerate(¸®½ºÆ®) = ¸®½ºÆ®ÀÇ ¿ä¼ÒµéÀ» a¿¡,ÀÎµ¦½º¸¦ b¿¡ ²¨³¿. (¸®½ºÆ®,|start=|1)·Î ÀÎµ¦½ºÀÇ ½ÃÀÛÀ» Á¤ÇÒ ¼ö ÀÖÀ½.
- Å¥ = deque(¸®½ºÆ®) ÀÚ·áÇü Á¦°ø. popleft()/appendleft()·Î ¿ÞÂÊ¿¡ ¿ä¼Ò »èÁ¦/Ãß°¡.
- È¸¹® ÆÇº°(µÚÁý±â) = s[::-1\], list(reversed(s)), "".join(reversed(s))

### list comprehension
- ¸®½ºÆ® Ç¥Çö½Ä
- [½Ä for º¯¼ö in ¸®½ºÆ®\] ÀÇ Çü½ÄÀ¸·Î »ç¿ë. [i for i in range(10)\] ·Î 0ºÎÅÍ 9±îÁöÀÇ ¸®½ºÆ®¸¦ »ý¼ºÇÒ ¼ö ÀÖ´Ù.
- [½Ä for º¯¼ö in ¸®½ºÆ® if Á¶°Ç½Ä\] ½ÄÀ¸·Î if ¹® µµ »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- for ¸¦ ¿©·¯¹ø ¾²¸é µÚ¿¡¼­ ºÎÅÍ Àû¿ëµÈ´Ù. 

## map,split
***
- a,b,c=map(int,input("Á¤¼ö ¼Â ÀÔ·Â : ").split())  # ¸®½ºÆ® ¹ÝÈ¯. ¸®½ºÆ®·Î º¯¼ö ¿©·¯°³ »ý¼º.
- map Àº ¸®½ºÆ®µî ½ÃÄö½º º¯È¯½Ã¸¸ »ç¿ë. L-value ¿©·¯°³. split()µµ ¿©·¯ ÀÚ·á¸¦ ÀÔ·ÂÇÒ¶§¸¸ »ç¿ë. ÇÏ³ªÀÇ ÀÚ·á¸¦ ÀÔ·Â¹ÞÀ»¶© input().
- a,b=input().split()

## range
***
- range(È½¼ö)
- range(½ÃÀÛ, ³¡), ³¡Àº ¹üÀ§¿¡ Æ÷ÇÔµÇÁö ¾ÊÀ½.
- range(½ÃÀÛ, ³¡, Áõ°¡Æø)

## tuple
***
- Æ©ÇÃ = (°ª, °ª, °ª)          # Æ©ÇÃ ¸¸µé±â
- Æ©ÇÃ = °ª, °ª, °ª            # °ýÈ£ ¾øÀÌ Æ©ÇÃ ¸¸µé±â
- Æ©ÇÃ = ()                    # ºó Æ©ÇÃ ¸¸µé±â
- Æ©ÇÃ = tuple()               # ºó Æ©ÇÃ ¸¸µé±â
- Æ©ÇÃ = tuple(list())         # tuple ¿¡ list()¸¦ ³Ö¾î¼­ ºó Æ©ÇÃ ¸¸µé±â
- Æ©ÇÃ = tuple(¸®½ºÆ®)         # tuple ¿¡ ¸®½ºÆ®¸¦ ³Ö¾î¼­ Æ©ÇÃ ¸¸µé±â
- Æ©ÇÃ = tuple(range(È½¼ö))    # range ·Î Æ©ÇÃ ¸¸µé±â
- Æ©ÇÃ = (°ª, )                # ¿ä¼Ò°¡ ÇÑ °³ÀÎ Æ©ÇÃ ¸¸µé±â
- Æ©ÇÃ = °ª,                   # ¿ä¼Ò°¡ ÇÑ °³ÀÎ Æ©ÇÃ ¸¸µé±â

##calc
***
- °ª in ½ÃÄö½º°´Ã¼             # ½ÃÄö½º °´Ã¼¿¡ Æ¯Á¤ °ªÀÌ ÀÖ´ÂÁö È®ÀÎ
- °ª not in ½ÃÄö½º°´Ã¼         # ½ÃÄö½º °´Ã¼¿¡ Æ¯Á¤ °ªÀÌ ¾ø´ÂÁö È®ÀÎ

- ½ÃÄö½º°´Ã¼1 + ½ÃÄö½º°´Ã¼2    # ½ÃÄö½º °´Ã¼¸¦ ¼­·Î ¿¬°áÇÏ¿© »õ ½ÃÄö½º °´Ã¼¸¦ ¸¸µê
- ½ÃÄö½º°´Ã¼ * Á¤¼ö            # ½ÃÄö½º °´Ã¼¸¦ Æ¯Á¤ È½¼ö¸¸Å­ ¹Ýº¹ÇÏ¿© »õ ½ÃÄö½º °´Ã¼¸¦ ¸¸µê
- Á¤¼ö * ½ÃÄö½º°´Ã¼            # ½ÃÄö½º °´Ã¼¸¦ Æ¯Á¤ È½¼ö¸¸Å­ ¹Ýº¹ÇÏ¿© »õ ½ÃÄö½º °´Ã¼¸¦ ¸¸µê

- len(½ÃÄö½º°´Ã¼)              # ½ÃÄö½º °´Ã¼ÀÇ ¿ä¼Ò °³¼ö(±æÀÌ) ±¸ÇÏ±â
- ½ÃÄö½º°´Ã¼[-À½¼ö\]            # ÀÎµ¦½º¸¦ À½¼ö·Î ÁöÁ¤ÇÏ¸é µÚ¿¡¼­ºÎÅÍ ¿ä¼Ò¿¡ Á¢±Ù, -1Àº µÚ¿¡¼­ Ã¹ ¹øÂ°
- del ½ÃÄö½º°´Ã¼[ÀÎµ¦½º\]       # ½ÃÄö½º °´Ã¼ÀÇ ¿ä¼Ò¸¦ »èÁ¦

##slice
***
- ½ÃÄö½º°´Ã¼[½ÃÀÛÀÎµ¦½º:³¡ÀÎµ¦½º\]                 # ÁöÁ¤µÈ ¹üÀ§ÀÇ ¿ä¼Ò¸¦ Àß¶ó¼­ »õ ½ÃÄö½º °´Ã¼¸¦ ¸¸µê
- ½ÃÄö½º°´Ã¼[½ÃÀÛÀÎµ¦½º:³¡ÀÎµ¦½º:ÀÎµ¦½ºÁõ°¡Æø\]      # ÀÎµ¦½º Áõ°¡ÆøÀ» ÁöÁ¤ÇÏ¸é ÇØ´ç °ª¸¸Å­ ÀÎµ¦½º¸¦ Áõ°¡½ÃÅ°¸é¼­ ¿ä¼Ò¸¦ °¡Á®¿È
- ½ÃÄö½º°´Ã¼[::Áõ°¡Æø\]                         # °´Ã¼ ÀüÃ¼¿¡¼­ Áõ°¡Æø¸¸Å­ ÀÎµ¦½º¸¦ Áõ°¡½ÃÅ°¸é¼­ ¿ä¼Ò¸¦ °¡Á®¿È
- del ½ÃÄö½º°´Ã¼[½ÃÀÛÀÎµ¦½º:³¡ÀÎµ¦½º\]            # Æ¯Á¤ ¹üÀ§ÀÇ ¿ä¼Ò¸¦ »èÁ¦(¿øº» °´Ã¼°¡ º¯°æµÊ)

### list,tuple
***
- ¸®½ºÆ®¸¦ ¸¸µç µÚ tuple(), ÅõÇÃÀ» ¸¸µçµÚ list()·Î ¼­·Î º¯È¯ °¡´É.
- ¸®½ºÆ®¿Í ÅõÇÃ ¾È¿¡ ¹®ÀÚ¿­À» ¸¸µé¸é ÇÑ ¹®ÀÚ¾¿ µé¾î°¡ »ý¼º.
- min(¹Ýº¹°¡´É °´Ã¼),max(¸®½ºÆ®µî),sum(½ÃÄö½º) = ÃÖ¼Ú°ª,ÃÖ´ñ°ª,ÇÕ°è
- ¸®½ºÆ® »Ó ¾Æ´Ï¶ó ÅõÇÃ¿¡µµ È°¿ë°¡´É. ¾Æ·¡ÀÇ °æ¿ì tuple(½Ä).
- [i for i in range(10)\], [i+5 for i in range(5)\], [i for i in range(10) if i%2==0\], [i * j for j in range(2, 10) for i in range(1, 10)\] µî 
  ¸®½ºÆ® Ç¥Çö½Ä¿¡¼­ ¹Ýº¹,Á¶°Ç¹® »ç¿ë °¡´É. Ã³¸®¼ø¼­´Â µÚ¿¡¼­ºÎÅÍ.

## list unpacking
***
- for x,y in [[10,20\],[30,40\]\] µîÀ¸·Î for ¹® ÇÑ¹ø¸¸ »ç¿ëÀÌ °¡´É. for µÚÀÇ º¯¼ö¿Í ¾ÈÂÊ ¸®½ºÆ®ÀÇ ¿ä¼ö ¼ö°¡ ÀÏÄ¡ÇØ¾ßÇÔ.
- for i in [[10,20,30\],[40,50,60\]\]: for j in i: ·Î ÁßÃ¸ for ¹® »ç¿ë °¡´É.
- [[0 for j in range(2)\] for i in range(3)\], [[0\] * 2 for i in range(3)\] ·Î 2Â÷¿ø ¸®½ºÆ® »ý¼º.
- [[0\] * i for i in [3, 1, 3, 2, 5\]\] ½ÄÀ¸·Î Åé´ÏÇü ¸®½ºÆ® »ý¼º.
- Á¤·ÄÀº sorted(¸®½ºÆ®, key=lambda student: student[±âÁØÀÌ µÉ ÀÎµ¦½º\], reverse=))·Î ¾ÈÂÊ ÀÎµ¦½º Á¤·Ä.

### list copy
***
- 2Â÷¿ø ¸®½ºÆ®¸¦ º¹»çÇÏ·Á¸é import copy/ º¯¼ö = copy.deepcopy(¸®½ºÆ®) ·Î º¹»ç.

## dictionary
***
- µñ¼Å³Ê¸® = {Å°1: °ª1, Å°2: °ª2}    # µñ¼Å³Ê¸® ¸¸µé±â
- µñ¼Å³Ê¸® = {}                      # ºó µñ¼Å³Ê¸® ¸¸µé±â
- µñ¼Å³Ê¸® = dict()                  # ºó µñ¼Å³Ê¸® ¸¸µé±â

- µñ¼Å³Ê¸®[Å°\]                       # µñ¼Å³Ê¸®¿¡¼­ Å°·Î °ª¿¡ Á¢±Ù
- µñ¼Å³Ê¸®[Å°\] = °ª                  # µñ¼Å³Ê¸®¿¡¼­ Å°¿¡ °ª ÇÒ´ç

- Å° in µñ¼Å³Ê¸®                     # µñ¼Å³Ê¸®¿¡ Æ¯Á¤ Å°°¡ ÀÖ´ÂÁö È®ÀÎ
- Å° not in µñ¼Å³Ê¸®                 # µñ¼Å³Ê¸®¿¡ Æ¯Á¤ Å°°¡ ¾ø´ÂÁö È®ÀÎ

- len(µñ¼Å³Ê¸®)                      # µñ¼Å³Ê¸®ÀÇ Å° °³¼ö(±æÀÌ) ±¸ÇÏ±â

- vars(µñ¼Å³Ê¸®) : °´Ã¼ÀÇ __dict\_\_¼Ó¼º ¹ÝÈ¯. ÇØ´ç °´Ã¼ÀÇ Å°¿Í value¸¦ ¸ðµÎ È®ÀÎ °¡´É.

### zip
***
- zip(a, b) > µ¿ÀÏÇÑ °³¼ö·Î ÀÌ·ç¾îÁø ÀÚ·áÇüÀ» ¹­¾îÁÜ.
- [1,2,3,4\]¿Í ['one','two','three','for'\]°¡ ÀÖ´Ù¸é [(1,'one'), ... \]
```python
# º¸Åë ÀÌ·±½ÄÀ¸·Î »ç¿ëÇÑ´Ù.
number_l=[1,2,3]
name_l=['one', 'two', 'three']
dic = dict()
for number , name in zip(number_l,name_l):
    dic[number] = name
```

### about dict
***
- µñ¼Å³Ê¸® : ¿¬°ü °ªÀ» ¹­¾î Á¦°øÇÏ´Â ¿ëµµ.(Å°-°ª ÇüÅÂÀÇ ÀÚ·áÇüÀ» ÇØ½Ã,ÇØ½Ã¸Ê,ÇØ½ÃÅ×ÀÌºíµîÀ¸·Î ºÎ¸£±âµµ ÇÔ)

- a={"a":97,"b":98,"c":99}ÀÇ Çü½ÄÀ¸·Î »ç¿ë. # Å° ÀÌ¸§ Áßº¹½Ã µÚÀÇ °ª¸¸ ÀúÀå.
- Å°´Â Á¤¼ö,½Ç¼ö,¹®ÀÚ¿­,ºÒµî °¡´É/ °ª¿¡´Â ¸®½ºÆ®µî ¸ðµç ÀÚ·áÇü »ç¿ë °¡´É.

- a=dict(a=97,b=98,c=99)(Å°=°ª)
- dict([(a,97),(b,98)\])(¸®½ºÆ® ¾È¿¡ (Å°,°ª))(Å°´Â ÀÚµ¿À¸·Î ¹®ÀÚ¿­·Î ¹Ù²ñ)
- a=dict(zip([a,b\],[97,98\]))¸®½ºÆ® È¤Àº Æ©ÇÃ·Î Å°,°ª
- dict({a:97,b:98})

- Å°ÀÇ °³¼ö=°ªÀÇ °³¼ö=len()À¸·Î ±¸ÇÒ ¼ö ÀÖÀ½.
- µñ¼Å³Ê¸®´Â Å°·Î Á¢±Ù #a["a"\]=65  # µñ¼Å³Ê¸®¿¡ ¾ø´Â Å° ÀÔ·Â ½Ã ÇÒ´ç ÈÄ Ãß°¡.

### dict calc
***
- µñ¼Å³Ê¸®.setdefault("Å°",°ª) = µñ¼Å³Ê¸®¿¡ ½Ö Ãß°¡. °ªÀÌ ¾øÀ¸¸é Å°¿¡ none.
- µñ¼Å³Ê¸®.update(Å°=°ª) = Å°ÀÇ °ª ¼öÁ¤. ¸¸¾à Å°°¡ ¾ø´Ù¸é »õ·Î Ãß°¡. Å°=°ª,Å°=°ªÀ¸·Î ¿©·¯°³ ÇÑ¹ø¿¡ ¼öÁ¤ °¡´É.
- update ´Â Å°°¡ ¹®ÀÚ¿­ÀÏ¶§¸¸ »ç¿ë°¡´É. Å°°¡ ¼ýÀÚ¸é (µñ¼Å³Ê¸®(¹Ù²Ü Å° : °ª))·Î ¼öÁ¤°¡´É. ()¾È¿¡´Â ¸®½ºÆ®, Æ©ÇÃµî ¹Ýº¹°¡´É°´Ã¼ ¸ðµÎ °¡´É. [[Å°1,°ª1\],[Å°2,°ª2\]\]ÇüÀ¸·Î ÀÌ·ïÁ®¾ßÇÔ.

- µñ¼Å³Ê¸®.pop(Å°,±âº»°ª) = Å°°¡ ÀÖÀ¸¸é Å°-°ª »èÁ¦ ÈÄ °ª ¹ÝÈ¯, ¾øÀ¸¸é ±âº»°ª ¹ÝÈ¯
- del µñ¼Å³Ê¸®["Å°"\]·Î Å° »èÁ¦.
- µñ¼Å³Ê¸®.popitem() = µñ¼Å³Ê¸® ¸¶Áö¸· °ª »èÁ¦ ÈÄ Æ©ÇÃ·Î ¹ÝÈ¯.
- µñ¼Å³Ê¸®.clear() = µñ¼Å³Ê¸® ¸ðµç °ª »èÁ¦.

- µñ¼Å³Ê¸®.get(Å°,±âº»°ª) = Å°°¡ ÀÖÀ¸¸é °ªÀ» °¡Á®¿À°í ¾øÀ¸¸é ±âº»°ª ¹ÝÈ¯. ±âÁ¸°ª ¾ø¾îµµ °¡´É.
- µñ¼Å³Ê¸®.items() .keys() .values() = Å°¿Í °ª, Å°µé, °ªµé °¡Á®¿È.

- º¯¼ö = dict.fromkeys(Å°µé,°ª) = Å°µé¿¡ ¸ðµÎ °ªÀ» Ã¤¿ö µñ¼Å³Ê¸® Á¦ÀÛ. °ªÀÌ ¾øÀ¸¸é None Ã¤¿ò.

- dict ³ª µñ¼Å³Ê¸®[Å°\]¿¡¼­ ¾ø´Â Å°¸¦ ¼±ÅÃÇÏ¸é ¿À·ù.
- ÀÌ ¿À·ù¸¦ ¾ø¾Ö±â À§ÇØ µñ¼Å³Ê¸® = defaultdict(ÀÚ·áÇü)À¸·Î ±âº»°ª »ý¼º.

### for dict
***
- for ¿¡¼­ µñ¼Å³Ê¸®¸¦ »ÌÀ¸¸é Å°¸¸ »ÌÈû.
- {Å°:°ª for Å°, °ª in µñ¼Å³Ê¸® if Á¶°Ç}
- {Å°:{Å°:°ª},Å°2:{Å°2:°ª2}}½ÄÀ¸·Î ÁßÃ¸ µñ¼Å³Ê¸®. Á¢±ÙÀº µñ¼Å³Ê¸®[¹Ù±ùÅ°\][¾ÈÂÊÅ°\]·Î ÇÔ.
- µñ¼Å³Ê¸®µµ ¸®½ºÆ®¿Í ¸¶Âù°¡Áö·Î ´ÜÀÏÀº µñ¼Å³Ê¸®.copy(), ÁßÃ¸Àº copy ¸ðµâÀÇ copy.deepcopy(µñ¼Å³Ê¸®)·Î º¹»çÇØ¾ß ¿ÏÀüº¹»çµÊ.
- µñ¼Å³Ê¸®¿¡¼­ Å°ÀÇ °³¼ö´Â len(µñ¼Å³Ê¸®)·Î ±¸ÇÒ ¼ö ÀÖÀ½.
- µñ¼Å³Ê¸®ÀÇ º´ÇÕÀº ,update(µñ¼Å³Ê¸®)³ª {**µñ¼Å³Ê¸®,**µñ¼Å³Ê¸®2}·Î °¡´ÉÇÔ.

# if
***
- if Á¶°Ç½Ä:
-    ÄÚµå
-    ¿©·¯ÁÙµµ °¡´É #Pass·Î »ý·«°¡´É.
- else: ·Î else,
- elif: ·Î if else »ç¿ë. ºó ¹®ÀÚ¿­, none,ºó ½ÃÄö½º,0µîÀº ÀüºÎ false.³ª¸ÓÁø true
- Á¶°Ç¿¡¼­ and,or »ç¿ë°¡´É. A and B. 0<=a<2°°ÀÌ ºÎµîÈ£¸¦ ¿¬´Þ¾Æ »ç¿ëÇÏ´Â°Íµµ °¡´É. & ,A not B, | µî ºÒ°¡. not A·Î »ç¿ë

- for º¯¼ö in range(È½¼ö(range ÀÇ ±â´É-Áõ°¡Æø µî-»ç¿ë °¡´É),º¯¼öµµ °¡´É)(±âÅ¸ ½ÃÄö½º ¦Ã¼µéµµ »ç¿ë °¡´É): reversed ·Î ¼ø¼­ µÚÁý±â °¡´É.
-    ÄÚµå     ·Î ¹Ýº¹¹® »ç¿ë.  ½ÃÄö½º °´Ã¼ÀÇ ÀÚ·áµéÀ» ÇÏ³ª¾¿ ²¨³» º¯¼ö¿¡ ´ëÀÔ ÈÄ ¹®Àå½ÇÇà.

- »ïÇ×¿¬»ê : ÂüÀÏ °æ¿ì if Á¶°Ç½Ä else °ÅÁþÀÏ °æ¿ì  ÀÇ Çü½ÄÀ¸·Î »ïÇ×¿¬»êÀÚ »ç¿ë°¡´É. 

# while
***
- i=0
- while i<100:
-    print("Hello")
-    i+=1    ·Î while ¹Ýº¹¹® »ç¿ë
-    break #continue

## rand
***
- import ·Î ¸ðµâÈ£Ãâ. random => random.random()À¸·Î ³­¼ö È£Ãâ. .randint(a,b)´Â aºÎÅÍ b »çÀÌÀÇ Á¤¼ö ·£´ý. .choice(½ÃÄö½º)´Â ½ÃÄö½º °´Ã¼ ¼Ó¿¡¼­ ·£´ý È£Ãâ.

# string calc
***
- ¹®ÀÚ¿­.replace("¹Ù²Ü ¹®ÀÚ¿­","»õ ¹®ÀÚ¿­") = ¹®ÀÚ¿­ º¯°æ ÈÄ °á°ú ¹ÝÈ¯.
- Å×ÀÌºí¸í = str.maketrans("¹Ù²Ü ¹®ÀÚ-µé-","»õ¹®ÀÚ-µé-")·Î Å×ÀÌºí »ý¼ºÈÄ ¹®ÀÚ¿­.translate(Å×ÀÌºí¸í)À¸·Î ¹®ÀÚº¯È¯.
- ¹®ÀÚ¿­.split() = °ø¹é ±âÁØÀ¸·Î ºÐ¸®ÇØ ¸®½ºÆ®È­. ¾È¿¡ ¹®ÀÚ¸¦ ³ÖÀ¸¸é ±âÁØ ¹®ÀÚ¿­´ë·Î ºÐ¸®.
- ±¸ºÐÀÚ¹®ÀÚ¿­.join(¹®ÀÚ¿­ ¸®½ºÆ®) = ±¸ºÐÀÚ ¹®ÀÚ¿­À» »çÀÌ¿¡ ³Ö¾î ¹®ÀÚ¿­µéÀ» ¿¬°á.
- ¹®ÀÚ¿­.upper(), ¹®ÀÚ¿­.lower() = ¹®ÀÚ¿­ ÀüºÎ ´ë¹®ÀÚ,¼Ò¹®ÀÚÈ­
- ¹®ÀÚ¿­.lstrip(), ¹®ÀÚ¿­.rstrip(), ¹®ÀÚ¿­.strip() = ¿ÞÂÊ,¿À¸¥ÂÊ,¾çÂÊ¿¡¼­ °ø¹éÁ¦°Å. (".")½ÄÀ¸·Î ¾È¿¡ ¹®ÀÚ¸¦ ³ÖÀ¸¸é ±× ¹®ÀÚ »èÁ¦. (".,")½ÄÀ¸·Î ¿©·¯ ¹®ÀÚ¸¦ ³ÖÀ¸¸é ¹®ÀÚ ÀüºÎ »èÁ¦.
- import string / .strip(string.punctuation)À¸·Î ±¸µÎÁ¡(´ëºÎºÐÀÇ ¹®Àå±âÈ£)»èÁ¦. °ø¹éµµ »èÁ¦ÇÏ°í ½ÍÀ¸¸é =" " ¶Ç´Â µÚ¿¡ .split()ÇÑ¹ø ´õ ºÙÀÌ±â(¸Þ¼­µå Ã¼ÀÌ´×, ¾Õ¿¡¼­ºÎÅÍ)

- ¹®ÀÚ¿­.ljust(±æÀÌ),rjust(±æÀÌ),center(±æÀÌ) : °ø¹éÀ» Æ÷ÇÔÇØ ±æÀÌ¸¸Å­ ¿ÞÂÊ,¿À¸¥ÂÊ,Áß°£¿¡ Á¤·Ä(¹èÄ¡). Áß°£ÀÇ °æ¿ì È¦¼ö¸é ¿ÞÂÊ¿¡ ÇÏ³ª ´õ.
- ¹®ÀÚ¿­.zfill(±æÀÌ) : ±æÀÌ¿¡ ¸Â°Ô ¿ÞÂÊ¿¡ 0À» Ã¤¿ò.
- ¹®ÀÚ¿­.find/rfind(Ã£À» ¹®ÀÚ¿­) : Ã£´Â ¹®ÀÚ¿­ÀÌ °¡Àå Ã³À½ ³ª¿Â °÷ÀÇ ÀÎµ¦½º ¹ÝÈ¯, ¾øÀ¸¸é -1 ¹ÝÈ¯. Ã£±â ½ÃÀÛÇÒ À§Ä¡¸¦ °°ÀÌ ÀÎÀÚ·Î ³ÖÀ» ¼ö ÀÖÀ½. 
- ¹®ÀÚ¿­.index/rindex(Ã£À» ¹®ÀÚ¿­) : find¿Í µ¿ÀÏÇÏ³ª ¾øÀ¸¸é ¿¡·¯.
- ¹®ÀÚ¿­.count("¹®ÀÚ¿­") : ¹®ÀÚ¿­¿¡¼­ ¹®ÀÚ¿­ÀÌ ³ª¿À´Â °³¼ö ¹ÝÈ¯.
- ¹®ÀÚ¿­.startswith(½ÃÀÛ¹®ÀÚ) : ¹®ÀÚ¿­ÀÌ Æ¯Á¤¹®ÀÚ·Î ½ÃÀÛÇÏ´ÂÁö ¿©ºÎ ¹ÝÈ¯. ½ÃÀÛÁöÁ¡À» ÀÎÀÚ·Î ÁÙ ¼öµµ ÀÖÀ½.
- ¹®ÀÚ¿­.endswith(³¡¹®ÀÚ) : ¹®ÀÚ¿­ÀÌ Æ¯Á¤ ¹®ÀÚ·Î ³¡³ª´ÂÁö ¿©ºÎ ¹ÝÈ¯. ¹®ÀÚ¿­ÀÇ ½ÃÀÛ°ú ³¡À» ÀÎÀÚ·Î ÁÙ ¼öµµ ÀÖÀ½.

- r"" : ¹®ÀÚ¿­À» raw¹®ÀÚ¿­·Î ÀÎ½ÄÇÏ°Ô ÇÔ. \°¡ \\·Î ÀÎ½ÄµÇ¾î Ãâ·ÂµÇ°Ô µÊ.


## format
***
- ¹®ÀÚ¿­(¾È¿¡ {ÀÎµ¦½º}).format(°ªµé) = ÀÎµ¦½º ºÎºÐ¿¡ format ºÎºÐ ¼Ó ÀÎµ¦½º¿¡ ¸Â´Â °ª »ðÀÔ. °°Àº ÀÎµ¦½º¸¦ ¿©·¯°³ ³Ö°Å³ª ÀÎµ¦½º¸¦ »ý·«ÇØµµ µÊ.
- ÀÎµ¦½º ´ë½Å format ¿¡ (ÀÌ¸§=°ª)À¸·Î ÁöÁ¤ ÈÄ {ÀÌ¸§}À» ³Ö¾îÁÖ´Â °Íµµ °¡´É.
- º¯¼ö¿¡ °ªÀ» ³Ö°í f¹®ÀÚ¿­(¼Ó¿¡ {º¯¼ö})·Î °£´Ü Æ÷¸ÅÆÃ °¡´É.
- {ÀÎµ¦½º(»ý·«°¡´É) : ¹æÇâ(<,>)±æÀÌ}.format(°ª)À¸·Î ljust,rjust,center Ã³·³ Á¤·Ä°¡´É.
- "%0°³¼öd"%¼ýÀÚ = "{ÀÎµ¦½º:0°³¼öd}".format(¼ýÀÚ)    "{ÀÎµ¦½º:%ÀÇ Æ¯¼ö¹®ÀÚ(02d,.3fµî)}".format()
- "%0°³¼ö.ÀÚ¸´¼öf"%¼ýÀÚ = "{ÀÎµ¦½º:0°³¼ö.ÀÚ¸´¼öf}".format(½Ç¼ö) ÀÌ¶§ ¼ýÀÚ°³¼ö´Â .°ú ±× ÀÌÇÏ¸¦ ¸ðµÎ Æ÷ÇÔÇÑ´Ù.

- "{ÀÎµ¦½º(Æ÷¸Ë¿¡¼­ÀÇ ÀÎµ¦½º):Ã¤¿ì±â(0,°ø¹éµî)|Á¤·Ä|±æÀÌ|.ÀÚ¸´¼ö|ÀÚ·áÇü}".format(,), "1.png"´Â"{0:03d}.{1}".format(int(x.split(".")[0]),x.split(".")[1])Ã³·³ »ç¿ë(1=±×´ë·Î).
- format(¼ýÀÚ, ','), '%±æÀÌs' % format(¼ýÀÚ, ',')(¿À¸¥ÂÊÁ¤·Ä), '{ÀÎµ¦½º:,}'.format(¼ýÀÚ) = Ãµ´ÜÀ§ ,±âÈ£ »ðÀÔ.

# set
***
- ¼¼Æ® = µñ¼Å³Ê¸®Ã³·³ {}¿¡ ÀúÀå. °¢ °ªÀº ,·Î ±¸ºÐ.
- ¼ø¼­°¡ Á¤ÇØÁ®ÀÖÁö ¾Ê¾Æ ¸Å¹ø ¼ø¼­°¡ ´Ù¸§. °ªÀÌ Áßº¹µÉ ¼ö ¾ø°í ÀÎµ¦½º¸¦ Áö¿øÇÏÁö ¾ÊÀ½.
- in, not in À¸·Î ¼¼Æ®¾È¿¡ °ªÀÌ ÀÖ´ÂÁö ¾ø´ÂÁö ÆÇ´Ü.
- set(¹®ÀÚ¿­µî ¹Ýº¹°¡´É °´Ã¼) ·Î ¼¼Æ® »ý¼º. Áßº¹µÇ´Â ¹®ÀÚ´Â ÇÏ³ª¸¸ µé¾î°¨. ºó¼¼Æ®´Â =set().
- set ´Â ¾È¿¡ set ¸¦ ³ÖÀ» ¼ö ¾ø°í, frozenset ¶ó´Â ¾ÈÀÇ ¿ä¼Ò º¯°æºÒ°¡ÀÇ ¼¼Æ®´Â frozenset ¸¦ ¾È¿¡ ³ÖÀ» ¼ö ÀÖÀ½.

## set calc
***
- ¼¼Æ®|¼¼Æ® , set.union(¼¼Æ®1,¼¼Æ®2) = ÇÕÁýÇÕ.
- ¼¼Æ®&¼¼Æ® , set.intersection(¼¼Æ®1,¼¼Æ®2) = ±³ÁýÇÕ.
- ¼¼Æ®-¼¼Æ® , set.difference(¼¼Æ®1,¼¼Æ®2) = Â÷ÁýÇÕ. (¾Õ¿¡¼­ µÚ¿Í °ãÄ¡´Â°É »­)
- ¼¼Æ®^¼¼Æ® , set.symmetric_difference(¼¼Æ®1,¼¼Æ®2) = ´ëÄªÀÚÁýÇÕ. (XOR)
- |= = .update() = ´õÇÔ.
- &= = .intersection_update() = °ãÄ¡´Â ¿ä¼Ò¸¸ ÀúÀå.
- -= = .difference_update() = ¾Õ¿¡¼­ µÚ¸¦ »«°Í¸¸ ÀúÀå.
- ^= = .symmetric_difference_update() = °ãÄ¡Áö ¾Ê´Â ¿ä¼Ò¸¸ ÀúÀå.
- <= = .issubset() = ¾Õ¿¡°Ô µÚÀÇ ºÎºÐÁýÇÕÀÎÁö È®ÀÎ. °ãÄ¡Áö ¾Ê´Â ÁøºÎºÐÁýÇÕÀº < .
- \>= = .issuperset() = ¾Õ¿¡°Ô µÚÀÇ »óÀ§ÁýÇÕÀÎÁö È®ÀÎ. °ãÄ¡Áö ¾Ê´Â Áø»óÀ§ÁýÇÕÀº >.
- == ,!= À¸·Î ¼­·Î °°ÀºÁö ´Ù¸¥Áö È®ÀÎ.
- .isdisjoint() = ¾Õ¿¡°Ô µÚ¿¡°Í°ú °ãÄ¡´Â°Ô ¾øÀ¸¸é Âü. ÀÖÀ¸¸é °ÅÁþ¹ÝÈ¯.

## set attribute
***
- .add(¿ä¼Ò) = ¼¼Æ®¿¡ ¿ä¼ÒÃß°¡.
- .remove(¿ä¼Ò) = ¿ä¼Ò Á¦°Å, ¾øÀ¸¸é ¿À·ù.
- .discard(¿ä¼Ò) = ¿ä¼ÒÁ¦°Å, ¾øÀ¸¸é ÆÐ½º.
- .pop() = ·£´ý¿ä¼Ò Á¦°Å ÈÄ ¹ÝÈ¯. ¿ä¼Ò°¡ ¾øÀ¸¸é ¿¡·¯.
- .clear() = ¸ðµç¿ä¼Ò »èÁ¦.
- len(¼¼Æ®) = ¼¼Æ®±æÀÌ¹ÝÈ¯.
- .copy() = ¼¼Æ® º¹»ç.
- ¼¼Æ® ¾È¿¡¼­µµ for, if °°Àº Ç¥Çö½Ä »ç¿ë °¡´É. if´Â ÂüÀÌ¿©¾ß Ãâ·Â.

# file
***
- ÆÄÀÏ°´Ã¼ = open("ÆÄÀÏÀÌ¸§","ÆÄÀÏ¸ðµå") > ÆÄÀÏ ¿­±â
- ÆÄÀÏ°´Ã¼.write("¹®ÀÚ¿­") > ÆÄÀÏ¿¡ ¾²±â. ¿©·¯ÁÙ ÀÛ¼ºÀº ¹Ýº¹¹® + \nÀ¸·Î °¡´É.
- º¯¼ö = ÆÄÀÏ°´Ã¼.read() > ÆÄÀÏ¿¡¼­ \[¹®ÀÚ¿­] ÀÐ±â
- ÆÄÀÏ°´Ã¼.close() > ÆÄÀÏ´Ý±â.
- with open("ÆÄÀÏÀÌ¸§", "ÆÄÀÏ¸ðµå") as ÆÄÀÏ°´Ã¼:
-   ÄÚµå >>> ½ÄÀ¸·Î·Î ÆÄÀÏ ¿ÀÇÂ ½Ã ´ÝÀ» ÇÊ¿ä ¾øÀ½.
- ÆÄÀÏ°´Ã¼.writelist(¸®½ºÆ®)´Â ¸®½ºÆ®¸¦ ÇÏ³ª¾¿ ÆÄÀÏ¿¡ ³ÖÀ½. \n ¾øÀ¸¸é ÇÑÁÙ·Î ³ÖÀ½.
- ÆÄÀÏ°´Ã¼.readlines()´Â ÆÄÀÏÀÇ ³»¿ëÀ» ÇÑ ÁÙ ¾¿ ¸®½ºÆ®·Î °¡Á®¿È.
- ÆÄÀÏ°´Ã¼.readline()Àº ÆÄÀÏÀ» ÇÑ ÁÙ¾¿ ÀÐÀ½. ´õÀÌ»ó ÀÐÀ» ÁÙÀÌ ¾øÀ¸¸é ""(ºó ¹®ÀÚ¿­)¹ÝÈ¯.
- for ¿¡ ÆÄÀÏ°´Ã¼¸¦ ÀúÀåÇÏ¸é ÆÄÀÏÀ» ÇÑ ÁÙ¾¿ ÀÐ¾î¿À°í, ÆÄÀÏÀÇ ÁÙ ¼ö¸¦ ¾Ë°íÀÖ´Ù¸é a,b,c=f ½ÄÀ¸·Î ¾ðÆÐÅ·ÇÏ´Â °Íµµ °¡´É ÇÔ.

## kind of open type
***
- r : ÀÐ±â¸ðµå. ÆÄÀÏÀÌ ¾øÀ¸¸é ¿¡·¯.
- r+ : ÀÐ±â ¶Ç´Â ¾²±â ¸ðµå. ÆÄÀÏÀÌ ¾øÀ¸¸é ¿¡·¯, ±âÁ¸ÆÄÀÏ À§¿¡ µ¤¾î¾¸.
- w : ¾²±â¸ðµå. ÆÄÀÏÀÌ ¾øÀ¸¸é »ý¼º.
- w+ : ÀÐ±â ¶Ç´Â ¾²±â ¸ðµå. ÆÄÀÏÀÌ ¾øÀ¸¸é »õ·Î»ý¼º. ÆÄÀÏ ÃÊ±âÈ­ ÈÄ ¾¸. 
- a : ÆÄÀÏ Ãß°¡ ¾²±â¸ðµå. ÆÄÀÏÀÌ ¾øÀ¸¸é »õ·Î »ý¼º. ÆÄÀÏÀÇ ³¡À¸·Î ÀÌµ¿ÇØ ¾¸.
- a+ : ÀÐ±â ¶Ç´Â ÆÄÀÏ Ãß°¡ ¸ðµå. ÆÄÀÏÀÌ ¾øÀ¸¸é »õ·Î »ý¼º.
- ?t : '?'¸ðµå¸¦ ÅØ½ºÆ® ¸ðµå·Î ¿®.
- ?b : '?'¸ðµå¸¦ ¹ÙÀÌ³Ê¸® ¸ðµå·Î ¿®.

## file import
***
- ÆÄÀÏ»ý¼ºÈÄ import ÆÄÀÏ¸í, ÆÄÀÏ¸í.ÆÄÀÏ¼Ó ÇÔ¼öÀÌ¸§() À¸·Î ±× ÆÄÀÏ¼Ó ÇÔ¼ö¸¦ °¡Á®¿Ã ¼ö ÀÖ´Ù. .º¯¼ö¸íÀ¸·Î º¯¼öµµ °¡Á®¿Ã ¼ö ÀÖ°í, 
  from ÇÔ¼ö¸í import ÆÄÀÏ¸íÀ¸·Î ±× ÇÔ¼ö¸¸ °¡Á®¿Ã ¼ö ÀÖÀ¸¸ç, ÀÌ·¸°Ô °¡Á®¿Â°Ç ÆÄÀÏ¸í¾øÀÌ ±×³É ÇÔ¼ö¸¸ ¾µ ¼ö ÀÖ´Ù.
- import ÆÄÀÏ¸í as º°¸íÀ¸·Î ÆÄÀÏ¸í´ë½Å º¯¸í.ÇÔ¼ö()·Î »ç¿ëÇÒ ¼ö ÀÖ´Ù.

## file pickle, glob
***
- pickle : ¸Þ¸ð¸®¿¡ ¿Ã¶ó°¡ ÀÖ´Â data ±× ÀÚÃ¼¸¦ dump, load ¸¦ ÅëÇØ ¿ÜºÎ¿¡ ÀúÀåÇÏ°í »ç¿ëÇÏ´Â °Í.
- import pickle , open("ÆÄÀÏ¸í","wb(write binary)")·Î ÆÄÀÏ ¿ÀÇÂ ÈÄ
- pickle.dump(°´Ã¼,ÆÄÀÏ¸í)À¸·Î ÆÄÀÏ¿¡ °´Ã¼ÀúÀå. ÇÑ ÁÙ ¾¿ ÀÔ·ÂµÊ. ÆÄÀÌ½ã¿¡ Á¸ÀçÇÏ´Â ¸ðµç°ÍÀÌ ÀÔ·Â°¡´ÉÇÔ.
- ¸¶Âù°¡Áö·Î import, open("ÆÄÀÏ¸í","rb")·Î ÆÄÀÏ ¿ÀÇÂ ÈÄ
- º¯¼ö=pickle.lode(ÆÄÀÏ°´Ã¼)·Î °´Ã¼·Îµå. ¿©·¯¹ø ¾²¸é ÇÑ ÁÙ ¾¿ °¡Á®¿È.

- from glob import glob
- glob(ÆÄÀÏ¸í) > ÀÎÀÚ·Î ¹ÞÀº ÆÐÅÏ°ú ÀÌ¸§ÀÌ ÀÏÄ¡ÇÏ´Â ¸ðµç ÆÄÀÏ°ú µð·ºÅÍ¸®ÀÇ ¸®½ºÆ®¸¦ ¹ÝÈ¯.

# function
***
- def ÇÔ¼ö¸í():
-   ÄÚµå  >>> ·Î ÇÔ¼ö »ý¼º. (a,b)½ÄÀ¸·Î º¯¼ö ÁöÁ¤µµ °¡´ÉÇÏ°í, return µµ »ç¿ë°¡´É. return ¿¡¼­ , (Æ©ÇÃÃ³¸®)·Î ¿©·¯°³ ¹ÝÈ¯ °¡´É. È¤Àº ¸®½ºÆ®µîµµ ¹ÝÈ¯°¡´É.
- def ¹Ù·Î ¹Ø¿¡ """ÀÌ·¸°Ô""" µ¶½ºÆ®¸µ(ÁÖ¼®)ÀÛ¼º°¡´É. ÇÔ¼ö¸í._doc_·Î µ¶½ºÆ®¸µ Ãâ·Â °¡´É.
- __ÇÔ¼ö(º¯¼öµµ µ¿ÀÏ)\__, ÇÔ¼ö >> public | _ÇÔ¼ö >> protected | __ÇÔ¼ö >> private.
- [¸Å°³º¯¼ö: ÀÚ·áÇü\] : ¸Å°³º¯¼ö¿¡ ÀÚ·áÇü ÁöÁ¤ °¡´É. list[ÀÚ·áÇü\]½ÄÀ¸·Î, Sequence ³»ºÎÀÇ ÀÚ·áÇü±îÁö ÁöÁ¤ °¡´É.
- [ÇÔ¼ö¸í() -> ÀÚ·áÇü\] : ÇÔ¼ö ¹ÝÈ¯ÀÚ·áÇü ÁöÁ¤ °¡´É.

## element
***
- À§Ä¡ÀÎ¼ö : ÇÔ¼ö¿¡ ÀÎ¼ö¸¦ ¼ø¼­´ë·Î ³Ö´Â ¹æ½Ä. ÀÎ¼ö¸¦ ¼ø¼­´ë·Î ³ÖÀ» ¶§´Â ¸®½ºÆ®³ª Æ©ÇÃÀ» »ç¿ëÇÒ ¼öµµ ÀÖÀ½. ¸®½ºÆ® ¶Ç´Â Æ©ÇÃ ¾Õ¿¡ *¸¦ ºÙ¿©¼­ ÇÔ¼ö¿¡ ³Ö¾îÁÖ¸é µÊ(¾ðÆÐÅ·).
 ´Ü °³¼ö°¡ ´Ù¸£¸é ¿À·ù ¹ß»ý.
- °¡º¯ÀÎ¼ö : ÇÔ¼ö ¼±¾ð½Ã ÇÔ¼ö¸í(*º¯¼ö¸í) À¸·Î ¼±¾ð(args). ÀÎ¼ö¸¦ ¸¶À½´ë·Î ³ÖÀ» ¼ö ÀÖÀ½(¾È ³Ö´Â°Íµµ °¡´É). ³ÖÀº ÀÎ¼öµéÀº Æ©ÇÃ·Î µé¾î°¨. 
  »ç¿ë½Ã¿£ for ¹® µîÀ¸·Î »ç¿ëÇÏ°í, ¸®½ºÆ®µîÀº ¾ðÆÐÅ· ÇÊ¿ä, °íÁ¤º¯¼ö¿Í °°ÀÌ ¾²¸é °íÁ¤ÀÌ Á¦ÀÏ ¾ÕÀ¸·Î ¿Í¾ß ÇÔ.
- Å°¿öµåÀÎ¼ö : ÇÔ¼ö È£Ãâ½Ã Å°¿öµå=°ª À¸·Î »ç¿ë. ÇÔ¼öÂÊ º¯¼ö¸í¸¸ ±â¾ïÇÏ¸é ¼ø¼­¸¦ ¸ÂÃßÁö ¾Ê¾Æµµ µÊ. sep,end µîµµ Å°¿öµå ÀÎ¼ö.
- µñ¼Å³Ê¸®ÀÇ Å°¸¦ Å°¿öµå¿Í °°°Ô ÇÑ ÈÄ (**µñ¼Å³Ê¸®)·Î µñ¼Å³Ê¸®ÀÇ °ªÀ» »ðÀÔ °¡´É. Å°´Â ¹«Á¶°Ç ¹®ÀÚ¿­ÀÌ¿©¾ßÇÏ¸ç ¸Å°³º¯¼öÀÇ ÀÌ¸§°ú Å° ÀÌ¸§, °ª½ÖÀÇ °³¼ö¿Í º¯¼öÀÇ °³¼ö°¡ °°¾Æ¾ß ÇÑ´Ù.
- Å°¿öµå °¡º¯ÀÎ¼ö : ÇÔ¼ö ¼±¾ð½Ã (**º¯¼ö¸í)À¸·Î ¼±¾ð(kwargs). ÀÔ·ÂÀº Å°¿öµå=°ª È¤Àº µñ¼Å³Ê¸® ¾ðÆÐÅ·, °á°ú¹°Àº µñ¼Å³Ê¸®. for °ú .item()µîÀ¸·Î Å°¿Í °ªÀ» »Ì¾Æ, 
  in À¸·Î °ªÀÌ ÀÖ´ÂÁö È®ÀÎ ÈÄ »ç¿ë. ´Ù¸¥ ÀÎ¼ö¿ÍÀÇ ¼ø¼­´Â °íÁ¤>°¡º¯>°¡º¯(Å°¿öµå).
- ÇÔ¼ö ¼±¾ð½Ã º¯¼ö¿¡ °ªÀ» ÇÒ´çÇÏ¸é ÃÊ±ê°ª. µû·Î ÁöÁ¤ÇÏÁö ¾ÊÀ¸¸é ÃÊ±ê°ªÀ¸·Î ³ª¿È. ÃÊ±ê°ªÀÌ ¾ø´Â º¯¼ö´Â Ç×»ó ÀÖ´Â º¯¼öº¸´Ù ¾Õ¿¡ ÀÖ¾î¾ß ÇÔ.

## lambda
***
- ¶÷´Ù(lambda)Ç¥Çö½Ä :  (ÇÔ¼ö°¡ µÉ)º¯¼ö = lambda ¸Å°³º¯¼ö(µé) : ½Ä  À¸·Î »ç¿ë. ¶÷´Ù½Ä ÀÚÃ¼¿¡ ()¸¦ ¾º¿ö »ç¿ëÇÒ ¼öµµ ÀÖ°í, ¶÷´Ù½ÄÀÇ ½Ä¿¡¼­´Â º¯¼ö¸¦ ¸¸µé¼ö´Â ¾øÀ¸³ª 
  ÀÌ¹Ì ÀÖ´Â º¯¼ö¸¦ »ç¿ëÇÒ ¼ö´Â ÀÖ´Ù.
- ¸Å°³º¯¼ö°¡ ¾ø´Â ¶÷´Ù½ÄÀº ±×³É :¸¸ ºÙÀÌ¸é µÇ°í, ÀÌ°ÍÀº ÇÔ¼ö¸¦ ÀÎ¼ö·Î »ç¿ëÇÒ ¶§ ÁÖ·Î »ç¿ëÇÑ´Ù(ex-map ÀÇ ÀÚ·áÇü ºÎºÐ¿¡ ÇÔ¼ö¸íÀ» ½áÁÙ¼öµµ ÀÖ´Âµ¥, ÀÌ¶§ ¶÷´Ù·Î ¾²¸é ÁÙ¼ö°¡ ÁÙ¾îµê).
- ¶÷´Ù½Ä¿¡¼­ Á¶°ÇºÎ Ç¥Çö½Ä »ç¿ë : lambda ¸Å°³º¯¼öµé: ½Ä1 if Á¶°Ç½Ä else ½Ä2 ÀÇ Çü½Ä. Á¶°ÇÀÌ ¸ÂÀ¸¸é ½Ä1,¾Æ´Ï¸é ½Ä 2·Î ¹ÝÈ¯. ¿©±â¿¡¼± else °¡ ÇÊ¼öÀÌ¸ç, elif ¸¦ »ç¿ëÇÒ ¼ö ¾ø´Ù.
- map(lambda x, y: x * y, a, b)Ã³·³ ¸ÊÀº ¹Ýº¹°¡´É º¯¼ö¸¦ ¿©·¯°³ ³ÖÀ» ¼ö ÀÖ°í, ÀÌ¶§´Â ¶÷´Ù½Ä¿¡ º¯¼ö¸¦ ¿©·¯°³ ³Ö¾îÁÖ¸é µÈ´Ù.
- filter(ÇÔ¼ö, ¹Ýº¹°¡´ÉÇÑ°´Ã¼)·Î »ç¿ë. ÇÔ¼öÀÇ ¹ÝÈ¯°ªÀÌ  True ÀÎ°Í¸¸ °¡Á®¿È. map Ã³·³ µû·Î ¸®½ºÆ®·Î °¨½ÎÁÖ°Å³ª ÇØ¾ßÇÔ. ¿©±â¿¡µµ ¶÷´Ù·Î and µîÀ» »ç¿ëÇØ ½áÁÖ¸é ÆíÇÔ.

## global, closer
***
- ÇÔ¼ö ¾È¿¡¼­ Àü¿ªº¯¼ö¸¦ ¼±¾ð,º¯°æÇÏ·Á¸é global ·Î ¼±¾ðÇØÁÖ¸é µÇ°í, ¹Ù±ùÂÊ Áö¿ªº¯¼ö¸¦ ¾ÈÂÊ¿¡¼­ º¯°æÇÏ·Á¸é nonlocal Å°¿öµå¸¦ »ç¿ëÇÏ¸é µÈ´Ù.

- ¾î¶² ÇÔ¼ö¿¡¼­ º¯¼ö¸¦ ¼±¾ðÇÏ°í, ±× ¾È¿¡¼­ ¶Ç ÇÔ¼ö¸¦ ¸¸µé¾î ±× ÀÚÃ¼¸¦ ¹ÝÈ¯ÇÏ°Å³ª ¶÷´Ù Ç¥Çö½ÄÀ» ¹ÝÈ¯ÇßÀ»¶§, ¹Ù±ù ÇÔ¼öÀÇ ¹ÝÈ¯°ªÀ» º¯¼ö¿¡ ÀúÀåÇÑ ÈÄ ±× º¯¼ö·Î ¾ÈÂÊ ÇÔ¼ö¸¦ »ç¿ëÇÏ¸é,
  ¹Ù±ùÂÊ ÇÔ¼ö ¾È¿¡ÀÖ´Â º¯¼ö¸¦ °è¼Ó ¾µ ¼ö ÀÖ´Ù.
- ÀÌ°Ô¹Ù·Î Å¬·ÎÀú. Å¬·ÎÀúÀÇ º¯¼ö¸¦ º¯°æÇÏ°í ½ÍÀ¸¸é nonlocal Å°¿öµå¸¦ »ç¿ë(¹Ù±ù°ú µ¿ÀÏÇÑ ÀÌ¸§¿¡´Ù nonlocal À» ºÙ¿© ¼±¾ð)ÇÏ¸é µÇ°í, º¸Åë ¶÷´ÙÇ¥Çö½Ä°ú °°ÀÌ ¾´´Ù.

# class
***
- class Å¬·¡½ºÀÌ¸§:
-  def ¸Þ¼­µå(self,±âÅ¸µîµî):
-    ÄÚµå
- Çü½ÄÀ¸·Î Å¬·¡½º»ý¼º. ÀÎ½ºÅÏ½º(º¯¼ö¸í)=Å¬·¡½º() ·Î ÀÎ½ºÅÏ½º»ý¼º.
- ÀÎ½ºÅÏ½º.¸Þ¼­µå()·Î ÀÎ½ºÅÏ½º ¸Þ¼­µå »ç¿ë. ¸Þ¼­µå ¾È¿¡¼­ Å¬·¡½º ¾ÈÀÇ ´Ù¸¥ ¸Þ¼­µå¸¦ È£ÃâÇÒ ¶§´Â self.¸Þ¼­µå()·Î È£ÃâÇÑ´Ù.
- isinstance(ÀÎ½ºÅÏ½º,Å¬·¡½º)·Î ÀÎ½ºÅÏ½º°¡ Å¬·¡½º¿¡ ¼ÓÇØÀÖ´ÂÁö È®ÀÎÇÑ´Ù. ¸ÂÀ¸¸é True,¾Æ´Ï¸é False ¹ÝÈ¯.

## class element
***
- Å¬·¡½º¿¡¼­ ¼Ó¼ºÀ» ¸¸µé¶§´Â
- def \_\_init__(self)://ÀÎ½ºÅÏ½º »ý¼º½Ã¸¶´Ù È£ÃâµÊ.
- self.¼Ó¼º=°ª   À¸·Î ÇÒ´çÇÔ. º¯¼ö»ý¼º°°ÀÌ ±× Å¬·¡½º ¾È¿¡¼­ ¾²°Å³ª, È£ÃâÇÏ¿© °ªÀ» ¹Ù²Ù°Å³ª ÇÒ ¼ö ÀÖ´Ù.

- def \_\_init__(self, name, age, address):
-     self.name = name
-     self.age = age
-     self.address = address
- Ã³·³ ¸Å°³º¯¼ö¸¦ ³Ö°í, ÃÊ±âÈ­ ÇØ ÀÎ½ºÅÏ½º¸¦ ¸¸µé¶§ °ªÀ» ¹ÞÀ» ¼ö ÀÖ´Ù. ÀÎ½ºÅÏ½º¸¦ ¸¸µé¶§ ¸Å°³º¯¼ö·Î health=health, mana=mana, 
  ability_power=ability_power Ã³·³ º¯¼ö=init ¼Ó ¸Å°³º¯¼ö¸¦ »ç¿ëÇØ º¯¼ö¸¦ Æ¯Á¤ ¸Å°³º¯¼ö¿¡ ³ÖÀ» ¼ö ÀÖ´Ù.
- ÀÌ¶§´Â a=Å¬·¡½º¸í(¸Å°³º¯¼öµé)Ã³·³ ÀÎ½ºÅÏ½º¸¦ »ý¼ºÇÑ´Ù.

- ¼Ó¼ºÁ¦ÀÛ½Ã¿¡ *args ·Î À§Ä¡ÀÎ¼ö(¸®½ºÆ®¸¦ ¾ðÆÐÅ·ÇØ¼­ ÀÎ½ºÅÏ½º »ý¼º½Ã ³ÖÀ½,self.name=args[0\]Ã³·³ °ª ÀúÀå),**kwargs ·Î Å°¿öµåÀÎ¼ö(µñ¼Å³Ê¸® ¾ðÆÐÅ· ¶Ç´Â [Å°¿öµå:ÀÎ¼ö\],
  self.name=kwargs["name"\]Ã³·³ °ª ÀúÀå)¸¦ »ç¿ëÇÒ ¼ö ÀÖ´Ù.

- Å¬·¡½º¸¦ pass ·Î Ã¤¿ö ºó Å¬·¡½º·Î ¸¸µç µÚ, ÀÎ½ºÅÏ½º¸¦ ¸¸µé°í °Å±â¿¡ ¾ø´Â ¼Ó¼º¿¡ °ªÀ» ÇÒ´çÇÏ¸é °è¼ÓÇØ¼­ ¼Ó¼ºÃß°¡¸¦ ÇÒ ¼ö ÀÖÁö¸¸, ÀÌ´Â ±× ÀÎ½ºÅÏ½º¿¡¸¸ Àû¿ëµÈ´Ù.
- __init__ÀÌ ¾Æ´Ñ ´Ù¸¥ ¸Þ¼­µå¿¡¼­ ¼Ó¼ºÀ» Ãß°¡ÇÏ¸é ±× ¸Þ¼­µå¸¦ È£ÃâÇØ¾ß¸¸ ¼Ó¼ºÀÌ »ý±ä´Ù.
- _\_slots\__ = \['¼Ó¼ºÀÌ¸§1, '¼Ó¼ºÀÌ¸§2']Ã³·³ ¸¸µé±â¸¦ Çã¿ëÇÏ°í ½ÍÀº ¼Ó¼º¸¸ ³ÖÀ¸¸é ´Ù¸¥ ¼Ó¼ºÀº »ý¼ºÀÌ Á¦ÇÑµÈ´Ù.
- self.__¼Ó¼º = ºñ°ø°³ ¼Ó¼º. Å¬·¡½º ¾È¿¡¼­¸¸(Å¬·¡½º¾ÈÀÇ ¸Þ¼­µå¿¡¼­¸¸ Á¢±Ù°¡´É, ±× ¸Þ¼­µå¸¦ È£ÃâÇØ¼­ »ç¿ë.) »ç¿ë°¡´É.
- def __¸Þ¼­µå ·Î ºñ°ø°³ ¸Þ¼­µå, __¼Ó¼º = °ªÀ¸·Î ºñ°ø°³ Å¬·¡½º ¼Ó¼ºµµ ¸¸µé ¼ö ÀÖÀ½.

## class global
***
- class Å¬·¡½ºÀÌ¸§:
-     ¼Ó¼º = °ª     Ã³·³ Å¬·¡½º ¹Ø¿¡ ¹Ù·Î ¼Ó¼ºÀ» ¸¸µé¾î Å¬·¡½º ¼Ó¼º(Å¬·¡½º¿¡ ¼ÓÇØÀÖÀ¸¸ç ¸ðµç ÀÎ½ºÅÏ½º¿¡¼­ °øÀ¯)À» ¸¸µé ¼ö ÀÖ´Ù. 
  Å¬·¡½º ¼Ó¼ºÀ» ¸Þ¼­µåµî¿¡¼­ »ç¿ëÇÏ·Á¸é Å¬·¡½º¸í.¼Ó¼º À¸·Î »ç¿ëÇÑ´Ù.
- ÆÄÀÌ½ãÀº ÀÎ½ºÅÏ½º, Å¬·¡½º ¼øÀ¸·Î Ã£±â ¶§¹®¿¡ °°Àº ÀÌ¸§ÀÌ ¾ø´Ù¸é ÀÎ½ºÅÏ½º.¼Ó¼º À¸·Î ½áµµ µÇÁö¸¸, ÀÌ·¯¸é °øÀ¯µÇ±âµµ ÇÏ°í ÀÇµµ¿Í ´Þ¶óÁú ¼ö ÀÖÀ¸´Ï Å¬·¡½º¸í.¼Ó¼ºÀ¸·Î »ç¿ë.
- Å¬·¡½º ¼Ó¼º: ¸ðµç ÀÎ½ºÅÏ½º°¡ °øÀ¯. ÀÎ½ºÅÏ½º ÀüÃ¼°¡ »ç¿ëÇØ¾ß ÇÏ´Â °ªÀ» ÀúÀåÇÒ ¶§ »ç¿ë
- ÀÎ½ºÅÏ½º ¼Ó¼º: ÀÎ½ºÅÏ½ºº°·Î µ¶¸³µÇ¾î ÀÖÀ½. °¢ ÀÎ½ºÅÏ½º°¡ °ªÀ» µû·Î ÀúÀåÇØ¾ß ÇÒ ¶§ »ç¿ë

### class deco
***
- Å¬·¡½º¿¡¼­ ¸Þ¼­µå¸¦ ¸¸µé ¶§ À§¿¡ @staticmethod ¸¦ ºÙ¿© Á¤Àû ¸Þ¼­µå·Î ¸¸µé¸é ¹Ù·Î Å¬·¡½º.¸Þ¼­µå()·Î È£ÃâÇÒ ¼ö ÀÖ´Ù. ÀÌ Á¤Àû¸Þ¼­µå´Â ÀÎ½ºÅÏ½º ¼Ó¼ºÀº »ç¿ë ºÒ°¡ÇÏ´Ù.
- ÀÎ½ºÅÏ½ºÀÇ ³»¿ëÀ» º¯°æÇØ¾ß ÇÒ ¶§´Â ÀÎ½ºÅÏ½º ¸Þ¼­µå, ÀÎ½ºÅÏ½º¿¡´Â º¯È­ ¾øÀÌ °á°ú¸¸ ±¸ÇÏ¸é µÉ ¶§¿¡´Â Á¤Àû ¸Þ¼­µå·Î »ç¿ëÇÑ´Ù. Á¤Àû ¸Þ¼­µå¿¡¼­´Â self ¸¦ ¸Å°³º¯¼ö·Î ³Ö¾îÁÖÁö ¾Ê¾Æµµ µÈ´Ù.

- ¸Þ¼­µå À§¿¡ @classmethod ¸¦ ºÙÀÌ¸é Å¬·¡½º ¸Þ¼­µå Á¦ÀÛ. Å¬·¡½º ¸Þ¼­µå´Â Ã¹¹øÂ° ¸Å°³º¯¼ö¿¡ cls ¸¦ ÁöÁ¤ÇØ¾ß ÇÔ. Å¬·¡½º ¸Þ¼­µå´Â ¸Þ¼­µå ¾È¿¡¼­ Å¬·¡½º ¼Ó¼º, 
  Å¬·¡½º ¸Þ¼­µå¿¡ Á¢±ÙÇØ¾ß ÇÒ ¶§ »ç¿ëÇÏ°í, cls ´Â Å¬·¡½º ÀÌ¹Ç·Î ¸Þ¼­µå ¾È¿¡¼­ ÀÎ½ºÅÏ½º = cls()³ª return cls()Ã³·³ ÇØ¼­ ÀÎ½ºÅÏ½º¸¦ ¸¸µé ¼ö ÀÖÀ½.
- @property ·Î getter, @ÇÔ¼ö(º¯¼ö)ÀÌ¸§.setter ·Î setter ¸¦ °£´ÜÈ÷ ±¸ÇöÇÒ ¼ö ÀÖ´Ù.

## class extends
***
- class ±â¹ÝÅ¬·¡½ºÀÌ¸§:
-   ÄÚµå
- class ÆÄ»ýÅ¬·¡½ºÀÌ¸§(±â¹ÝÅ¬·¡½ºÀÌ¸§):
-   ÄÚµå   ·Î Å¬·¡½º¸¦ »ó¼Ó½ÃÅ³ ¼ö ÀÖ´Ù. ÀÌ·¯¸é ÆÄ»ýÅ¬·¡½º¿¡¼­µµ ±â¹ÝÅ¬·¡½ºÀÇ ¸ðµç ¿ä¼ÒµéÀ» »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- Å¬·¡½º »ó¼ÓÀº ¿¬°üµÇ¸é¼­ µ¿µîÇÑ ±â´ÉÀÏ ¶§, °°Àº Á¾·ùÀÌ¸ç µ¿µîÇÑ °ü°èÀÏ¶§ »ç¿ëÇÑ´Ù. ¿µ¾î·Î´Â is-a °ü°è¶ó°í ÇÔ.
- issubclass(ÆÄ»ýÅ¬·¡½º, ±â¹ÝÅ¬·¡½º)·Î ±â¹ÝÅ¬·¡½ºÀÇ ÆÄ»ýÅ¬·¡½º°¡ ¸Â´ÂÁö È®ÀÎ ÇÒ ¼ö ÀÖ´Ù. boolean À¸·Î ¹ÝÈ¯.

- ÆÄ»ýÅ¬·¡½º¿¡¼­ ±â¹ÝÅ¬·¡½ºÀÇ ¼Ó¼ºÀ» »ç¿ëÇÏ·Á¸é ±â¹ÝÅ¬·¡½ºÀÇ __init__¸Þ¼­µå¸¦ È£ÃâÇØÁà¾ß ÇÑ´Ù. ±×·¯Áö ¾ÊÀ¸¸é ¼Ó¼ºÀÌ »ý¼ºµÇÁö ¾Ê±â ¶§¹®ÀÌ´Ù.
- ÀÌ¶§´Â ÆÄ»ýÅ¬·¡½º¿¡¼­ super().\_\_init__() À¸·Î ±â¹ÝÅ¬·¡½ºÀÇ __init__¸Þ¼­µå¸¦ È£ÃâÇÑ´Ù.
- ÆÄ»ýÅ¬·¡½º¿¡¼­ __init__¸Þ¼­µå¸¦ »ç¿ëÇÏÁö ¾Ê¾Ò´Ù¸é ÀÚµ¿À¸·Î ±â¹ÝÅ¬·¡½ºÀÇ __init__ÀÌ È£ÃâµÇ¹Ç·Î super()À» »ç¿ëÇÒ ÇÊ¿ä ¾ø´Ù.
- super(ÆÄ»ýÅ¬·¡½º, self).¸Þ¼­µå ·Î ÇöÀçÅ¬·¡½º¸¦ ¸íÈ®ÇÏ°Ô Ç¥½ÃÇÒ ¼öµµ ÀÖ´Ù. ÀÌ °æ¿ì ±â´É¿¡´Â Â÷ÀÌ°¡ ¾ø´Ù.

## override
***
- ¸Þ¼­µå ¿À¹ö¶óÀÌµù: ÆÄ»ýÅ¬·¡½º¿¡¼­ ±â¹ÝÅ¬·¡½ºÀÇ ¸Þ¼­µå¸¦ ÀçÁ¤ÀÇ. ¿ø·¡±â´ÉÀ» À¯ÁöÇÏ¸é¼­ »õ·Î¿î ±â´ÉÀ» µ¡ºÙÀÏ¶§ »ç¿ë.
- def greeting(self):
-  super().greeting() Çü½ÄÀ¸·Î »ç¿ë. ÀÌ µÚ¿¡ ´Ù¸¥ ÄÚµå¸¦ µ¡ºÙÀÌ´Â ½ÄÀ¸·Î(ÀÌ °æ¿ì, ±â¹ÝÅ¬·¡½ºÀÇ ÀÎ»ñ¸» ¾È³çÇÏ¼¼¿ä¸¦ Ãâ·ÂÇÏ°Ô ÇÑ µÚ ´Ù¸¥ ÀÎ»ñ¸»À» Ãß°¡ÇßÀ½) »ç¿ëÇÑ´Ù.

## multiple extends
***
- class ÆÄ»ýÅ¬·¡½ºÀÌ¸§(±â¹ÝÅ¬·¡½ºÀÌ¸§1, ±â¹ÝÅ¬·¡½ºÀÌ¸§2): Çü½ÄÀ¸·Î ´ÙÁß»ó¼ÓÀ» ÇÒ ¼ö ÀÖ´Ù.
- Å¬·¡½º A¸¦ »ó¼Ó¹Þ¾Æ¼­ B, C¸¦ ¸¸µé°í, Å¬·¡½º B¿Í C¸¦ »ó¼Ó¹Þ¾Æ¼­ D¸¦ ¸¸µå´Â Çü½ÄÀ» ´ÙÀÌ¾Æ¸óµå »ó¼ÓÀÌ¶ó°í ÇÏ´Âµ¥, 
  ÀÌ °æ¿ì À§ÀÇ ¼Â ¸ðµÎ °°Àº ÀÌ¸§ÀÇ ¸Þ¼­µå¸¦ °¡Áö°í ÀÖ´Ù¸é ¾î¶² ¸Þ¼­µå¸¦ È£ÃâÇØ¾ß ÇÒÁö ¾Ö¸ÅÇØÁü.
- ÀÌ °æ¿ì ÆÄÀÌ½ã¿¡¼­´Â ¸Þ¼­µå Å½»ö¼ø¼­¸¦ µû¸£´Â µ¥, ÀÌ°Ç Å¬·¡½º.mro()(Å¬·¡½º.\_\_mro__ Çü½Äµµ °°Àº ³»¿ë)·Î È®ÀÎÈú ¼ö ÀÖ´Ù. 
  \[<class '\_\_main__.D'>, <class '\_\_main__.B'>, <class '\_\_main__.C'>, <class '\_\_main__.A'>, <
  class 'object'>]Çü½ÄÀ¸·Î Ãâ·ÂµÊ.(object Å¬·¡½º´Â ¸ðµç Å¬·¡½ºÀÇ Á¶»ó. ¸ðµç Å¬·¡½º´Â ÀÌ Å¬·¡½º¸¦ »ó¼Ó¹ÞÀ½.)
- Áï, »ó¼Ó ´Ü°è°¡ °¡±î¿î °Í ºÎÅÍ, ¿ÞÂÊ¿¡¼­ ¿À¸¥ÂÊ ¼ø¼­·Î Å½»öÇÔ.

## abstract class
***
- Ãß»óÅ¬·¡½º:from abc import *·Î abc ¸ðµâÀÇ ¸ðµç Å¬·¡½º¿Í ¸Þ¼­µå¸¦ °¡Á®¿Â ÈÄ(abc.ABCMeta, @abc.abstractmethod »ç¿ë)
- class Ãß»óÅ¬·¡½ºÀÌ¸§(metaclass=ABCMeta):
-  @abstractmethod
-  def ¸Þ¼­µåÀÌ¸§(self):
-   pass(Ãß»ó¸Þ¼­µå´Â Á÷Á¢ È£ÃâµÉ ÀÏÀÌ ¾ø±â ¶§¹®¿¡ ºó ¸Þ¼­µå·Î ¸¸µê)·Î »ç¿ë. »ó¼Ó¹Þ´Â Å¬·¡½º¿¡¼­ Å¬·¡½ºÀÇ Á¤ÀÇ¸¦ °­Á¦ÇÏ±â À§ÇØ »ç¿ë. Ãß»óÅ¬·¡½ºÀÇ ¸ðµç Ãß»ó ¸Þ¼­µå¸¦ ±¸ÇöÇØ¾ßÇÔ.


### has a class
***
- µ¿µîÇÑ Å¬·¡½º°¡ ¾Æ´Ï¶ó ¾î¶² Å¬·¡½ºµéÀ» °ü¸®ÇÏ´Â Å¬·¡½º¸¦ ¸¸µå·Á¸é ¸®½ºÆ® ¼Ó¼º¿¡ self.person_list.append(person)½ÄÀ¸·Î ÀÎ½ºÅÏ½ºµéÀ» ³Ö¾î °ü¸®ÇÑ´Ù.
- ÀÌ °æ¿ì, PersonList ´Â Person À» °¡Áö°í ÀÖ´Ù°í ÇÒ ¼ö ÀÖÀ¸¸ç, ÀÌ·± °ü°è¸¦ Æ÷ÇÔ°ü°è, ¿µ¾î·Î´Â has-a °ü°è¶ó°í ÇÑ´Ù.

### about class
***
- Math ÆÐÅ°Áö(ÆÄÀÌ½ã ³»Àå) > sqrt(°ª):Á¦°ö±Ù | pow(°ª,Áö¼ö):°ªÀÇ Áö¼öÁ¦°ö ¹ÝÈ¯ | abs(°ª):Àý´ñ°ª ¹ÝÈ¯
- (collections ¸ðµ¨ import)Å¬·¡½º¸í = collections.namedtuple('ÀÚ·áÇüÀÌ¸§', ['¿ä¼ÒÀÌ¸§1', '¿ä¼ÒÀÌ¸§2'\])·Î °¢ ¿ä¼Ò¿¡ ÀÌ¸§À» ÁöÁ¤ÇØÁÖ´Â namedtuple »ý¼º.
- ÀÎ½ºÅÏ½º = Å¬·¡½º(°ª1,°ª2)/Å¬·¡½º(¿ä¼Ò1=°ª1,¿ä¼Ò2=°ª2)·Î ÀÎ½ºÅÏ½º »ý¼º, ÀÎ½ºÅÏ½º.¿ä¼Ò1/ÀÎ½ºÅÏ½º[ÀÎµ¦½º\]·Î ¿ä¼Ò Á¢±Ù. Å¬·¡½º¸¦ µû·Î ¸¸µé°í 
  __init__À¸·Î ¿ä¼Ò¸¦ ¸¸µé±âº¸´Ù ¿ä¼Ò¸¸ ¾²´Âµ¥ À¯¿ëÇÑµí.
- ¹Í½ºÀÎ : ´Ù¸¥ Å¬·¡½º¿¡¼­ »ç¿ëÇÒ ¼ö ÀÖµµ·Ï °øÅëÀûÀÎ ¸Þ¼­µå¸¦ ¸ð¾Æ ³õÀº Å¬·¡½º. HelloMixIn °°Àº ¹æ½ÄÀ¸·Î »ç¿ë.

# try
***
- try:
-   ½ÇÇàÇÒ ÄÚµå
- except:
-   ¿¹¿Ü°¡ ¹ß»ýÇßÀ» ¶§ Ã³¸®ÇÏ´Â ÄÚµå ·Î ¿¹¿ÜÃ³¸®. try ¿¡¼­ ¿À·ù ¹ß»ý½Ã ¹Ù·Î expect ·Î °¨.
- expect ¿¹¿ÜÀÌ¸§: À¸·Î Æ¯Á¤ ¿¹¿Ü¸¸ Ã³¸®. expect ¸¦ ¿©·¯°³ »ç¿ëÇØ¼­ Æ¯Á¤ ¿¹¿Ü¿¡´Â Æ¯Á¤ ÄÚµå¸¦ »ç¿ëÇÏ°Ô ÇÒ ¼ö ÀÖÀ½.
- expect ¿¹¿Ü as º¯¼ö(ÁÖ·Î e): ·Î º¯¼ö¿¡ ¿¹¿Ü¸Þ¼¼Áö¸¦ ¹Þ¾Æ¿Ã ¼ö ÀÖÀ½.
- expect Exception as º¯¼ö: ·Î ¸ðµç ¿¹¿ÜÀÇ ¿¡·¯ ¸Þ¼¼Áö¸¦ ¹Þ¾Æ ¿Ã ¼ö ÀÖÀ½.
- ¿¹¿Ü°¡ ¿©·¯°³ ¹ß»ýÇÏ¸é ¸ÕÀú ¹ß»ýÇß°Å³ª ³ôÀº °èÃþÀÇ ¿¹¿Ü ¸Þ¼¼Áö°¡ Ãâ·ÂµÈ´Ù.

- try: ÀÌ°ÍµéÀº ÇÔ¼ö°¡ ¾Æ´Ï´Ï try ¾È¿¡¼­ ¸¸µç º¯¼öµµ ¹Ù±ù¿¡¼­ »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- expect:
- else: ·Î ¿¹¿Ü°¡ ¹ß»ýÇÏÁö ¾Ê¾ÒÀ»¶§ ½ÇÇàÇÒ ÄÚµå¸¦ ÁöÁ¤ÇÒ ¼ö ÀÖÀ½. expect »ý·« ºÒ°¡.
- finally: ·Î ¿¹¿Ü ¿©ºÎ¿Í´Â »ó°ü¾øÀÌ Ç×»ó ½ÇÇàÇÒ ÄÚµå¸¦ ÁöÁ¤ÇÒ ¼ö ÀÖ´Ù. expect,else »ý·« °¡´É.

# raise
***
- raise ¿¹¿ÜÀÌ¸§('¿¡·¯¸Þ½ÃÁö') ·Î ¿¹¿Ü¸¦ ¹ß»ý½ÃÅ³ ¼ö ÀÖ´Ù.
  (ifÀý ¾È¿¡ ³Ö¾î¼­ ¿¡·¯¸¦ ¹ß»ý½ÃÅ°´Â µî, ¿¡·¯¸íÀº ÀÌ¹Ì ÀÖ´Â ¿¡·¯¸íÀ¸·Î. ±× ¿¡·¯¸¦ ¿¡·¯¸Þ¼¼Áö¿Í ¹ß»ý»óÈ²¸¸ ´Ù¸£°Ô ÇØ ¹ß»ý½ÃÅ°´Â °Í.)
- raise ¸¦ try ¹Û¿¡¼­ »ç¿ëÇÏ¸é ±× ÄÚµåºí·Ï ¾È¿¡¼­ except ¸¦ Ã£¾Æ ½ÇÇàÇÏ°í, except °¡ ¾øÀ¸¸é ±×´ë·Î ±×³É ¿À·ù°¡ ¹ß»ýÇØ ½ÇÇàÀÌ Á¾·áµÈ´Ù.
- except:
- raise ·Î ÇöÀç ¿¹¿Ü¸¦ ´Ù½Ã ¹ß»ý½ÃÄÑ »óÀ§ ÄÚµåºí·Ï(ÇÔ¼ö¸é ÇÔ¼ö ¹ÛÀÇ expect ¿¡¼­ °°ÀÌ)ÀÇ except ¿¡¼­ ¿¹¿ÜÃ³¸®.
- raise ¿¹¿ÜÀÌ¸§("¿¹¿Ü¸Þ¼¼Áö")·Î ´Ù¸¥ ¿¹¿Ü¸¦ ÁöÁ¤ÇÏ°í ¿¡·¯¸Þ¼¼Áö¸¦ ³ÖÀ» ¼öµµ ÀÖÀ½.

## assert
***
- assert Á¶°Ç½Ä/assert Á¶°Ç½Ä,¿¡·¯¸Þ¼¼Áö ·Î Á¶°Ç½ÄÀÌ °ÅÁþÀÌ¸é AssertionError ¹ß»ý,ÂüÀÌ¸é ±×³É ³Ñ±æ ¼ö ÀÖ´Ù. ÁÖ·Î ³ª¿Í¼± ¾È µÇ´Â Á¶°ÇÀ» °Ë»çÇÒ ¶§ »ç¿ëÇÑ´Ù.
- assert ´Â µð¹ö±ë¸ðµå¿¡¼­¸¸ ½ÇÇàµÇ¸ç(ÆÄÀÌ½ã ±âº»ÀÌ µð¹ö±ë¸ðµå), ½ÇÇàµÇÁö ¾Ê°Ô ÇÏ·Á¸é python -O ½ºÅ©¸³Æ®ÆÄÀÏ.py Ã³·³ ½ÇÇà½ÃÅ°¸é µÈ´Ù.

### make Exception
***
- class ¿¹¿ÜÀÌ¸§(Exception):
-  def \_\_init__(self):
-   super().\_\_init__('¿¡·¯¸Þ½ÃÁö') ·Î Á÷Á¢ ¿¹¿Ü¸¦ ¸¸µé ¼ö ÀÖ´Ù.

# iter
***
- ÀÌÅÍ·¹ÀÌÅÍ(¹Ýº¹ÀÚ) : °ªÀ» Â÷·Ê´ë·Î ²¨³¾ ¼ö ÀÖ´Â °´Ã¼
- ÀÌÅÍ·¹ÀÌÅÍ´Â Â÷·Ê´ë·Î ¹Ýº¹ÇÏ´Ù ¿¹¿Ü¸¦ ¹ß»ý½ÃÄÑ ¹Ýº¹À» ³¡³½´Ù.
- dir(°´Ã¼):°´Ã¼ÀÇ ¸Þ¼­µå È®ÀÎ. ¸Þ¼­µåÁß __iter__ÀÌ ÀÖÀ¸¸é ¹Ýº¹°¡´É °´Ã¼.
- it = \[1, 2, 3].\_\_iter__() Ã³·³ º¯¼ö¿¡ ¸®½ºÆ®ÀÇ ÀÌÅÍ·¹ÀÌÅÍ¸¦ ÀúÀåÇÑ ÈÄ it.\_\_next__()°°ÀÌ \_\_next__()·Î Â÷·Ê´ë·Î ²¨³¾ ¼ö ÀÖÀ½.
- ¸¶Áö¸· ¿ä¼Ò ÀÌÈÄ \_\_iter__()»ç¿ë½Ã StopIteration ¿¹¿Ü°¡ ¹ß»ýÇÔ.

- ½ÃÄö½º°´Ã¼ :¿ä¼ÒÀÇ ¼ø¼­°¡ Á¤ÇØÁ®ÀÖ°í, ¿¬¼ÓÀû. ¹Ýº¹°¡´É °´Ã¼¿¡ ¼ÓÇÔ. ¸®½ºÆ®,Æ©ÇÃ,¹®ÀÚ¿­,range.
- ¹Ýº¹°¡´É°´Ã¼ :¿ä¼ÒÀÇ ¼ø¼­¿Í´Â »ó°ü ¾øÀÌ ¿ä¼Ò¸¦ ÇÑ ¹ø¿¡ ÇÏ³ª¾¿ ²¨³¾ ¼ö ÀÖÀ½. ½ÃÄö½º°´Ã¼+µñ¼Å³Ê¸®,¼¼Æ®.
- ÀÌÅÍ·¹ÀÌÅÍ : \_\_next__ ¸Þ¼­µå¸¦ »ç¿ëÇØ¼­ Â÷·Ê´ë·Î °ªÀ» ²¨³¾ ¼ö ÀÖ´Â °´Ã¼. ÀÌÅÍ·¯ºí °´Ã¼¿¡¼­ \_\_iter__()¸¦ »ç¿ëÇØ ÀÌÅÍ·¹ÀÌÅÍ·Î ¹Ù²Ü ¼ö ÀÖÀ½.
- ÀÌÅÍ·¯ºí : ¹Ýº¹ °¡´ÉÇÑ °´Ã¼. \_\_next__ ¾øÀÌ \_\_iter__¸¸ °¡Áö°í ÀÖÀ½. ¸®½ºÆ®,¼¼Æ®,¹®ÀÚ¿­ µî. for¹®ÀÇ °æ¿ì ÀÌÅÍ·¯ºí°´Ã¼¿¡¼­ ÀÌÅÍ·¹ÀÌÅÍ¸¦ ¾ò¾î °ªÀ» ÇÏ³ª¾¿ ²¨³»¿È.

- iter(ÀÌÅÍ·¯ºí °´Ã¼,¹Ýº¹À» ³¡³¾°ª), iter(lambda : random.randint(0, 5), 2) µî iter() ¸Þ¼­µå »ç¿ë°¡´É
- next(ÀÌÅÍ·¯ºí °´Ã¼, ±âº»°ª)À¸·Î ¹Ýº¹ÇÒ ¼ö ÀÖÀ» ¶§´Â ÇØ´ç °ªÀ» Ãâ·ÂÇÏ°í, ¹Ýº¹ÀÌ ³¡³µÀ» ¶§´Â ±âº»°ªÀ» Ãâ·ÂÇÏ°Ô ÇÒ ¼ö ÀÖÀ½.

## make iter
***
- Å¬·¡½º¿¡ __iter__°ú __next__¸¦ ¸ðµÎ ±¸ÇöÇÏ°Å³ª __getitem__À» ±¸ÇöÇÏ¸é ÀÌÅÍ·¹ÀÌÅÍ¸¦ ¸¸µé ¼ö ÀÖÀ½(ÀÌÅÍ·¹ÀÌÅÍ ÇÁ·ÎÅäÄÝ Áö¿ø).

- \_\_iter__: return self Ã³·³ ÀÚ±â¸¦ ±×´ë·Î ¸®ÅÏÇÔ.
- \_\_next__: __init__¿¡¼­ ¹ÞÀº stop °ú Á¤ÀÇÇÑ current ¸¦ ÀÌ¿ëÇØ
- self.current += 1
-  if self.current < self.stop:
-   return self.current
-  else:
-   raise StopIteration Ã³·³ ¸¸µé¸é µÈ´Ù. if self.current * self.multiple < self.stop: Ã³·³ ´Ù¸¥ º¯¼ö¸¦ ÀÌ¿ëÇØµµ µÈ´Ù.

- \_\_getitem__(self, ÀÎµ¦½º): ·Î ÀÎµ¦½º·Î Á¢±ÙÇÒ ¼ö ÀÖ´Â ÀÌÅÍ·¹ÀÌÅÍ »ý¼º.
-        if index < self.stop:
-            return index   #index¿Í °°Àº ¼ö¸¦ ¹ÝÈ¯. ¾Æ´Ï¸é ´Ù¸¥ ½ÄÀ» ³Ö¾îµµ µÈ´Ù.
-        index+=1
-        if index<self.stop-self.start: //ÀÎµ¦½º ÁöÁ¤ÀÌ ¾øÀ¸¸é 0ºÎÅÍ ½ÃÀÛÀÎµí.
-            return "{0:02d}:{1:02d}:{2:02d}".format(((self.time+index)//3600),((self.time+index)%3600//60),((self.time+index)%3600%60)) °°Àº½Ä.
-        else: raise IndexError Ã³·³ ÄÚµå¸¦ Â¥¸é µÈ´Ù. »ç¿ëÀº Counter(3)[2]³ª for i in Counter(3): Ã³·³ ÇÏ¸é µÈ´Ù.

# generator
***
- Á¦³Ê·¹ÀÌÅÍ:ÀÌÅÍ·¹ÀÌÅÍ »ý¼º ÇÔ¼ö. ÀÌÅÍ·¹ÀÌÅÍº¸´Ù ÈÎ¾À °£´ÜÇÏ°Ô ÀÛ¼º °¡´É.
- yield °ª À» ¿¬¼ÓÇØ¼­ ½á ÀÌÅÍ·¹ÀÌÅÍ¸¦ ¸¸µé ¼ö ÀÖÀ½. yield(»ý»êÇÏ´Ù,¾çº¸ÇÏ´Ù.): °ªÀ» ÇÔ¼ö ¹Ù±ùÀ¸·Î Àü´ÞÇÏ¸ç ÄÚµå ½ÇÇàÀ» ÇÔ¼ö ¹Ù±ù¿¡ ¾çº¸ÇÏ°í,
  Àü´ÞÇÑ °ªÀº next ÀÇ ¹ÝÈ¯°ªÀ¸·Î ³ª¿È. Á¦³Ê·¹ÀÌÅÍ¿¡¼­ return ¿¡ ¹ÝÈ¯°ªÀ» ÁÖ¸é StopIteration ¿¹¿Ü¿¡ ¿¡·¯¸Þ¼¼Áö·Î µé¾î°¨.
- yield i.upper()Ã³·³ ÇÔ¼ö¸¦ È£ÃâÇÏ¸é ±× ÇÔ¼öÀÇ ¹ÝÈ¯°ªÀ» Àü´ÞÇÔ.
- yield from ¹Ýº¹°¡´É°´Ã¼ ·Î ¹Ýº¹°¡´É°´Ã¼ÀÇ ¿ä¼ÒµéÀ» ÇÏ³ª¾¿ ¹ÛÀ¸·Î Àü´Þ.
- ¸®½ºÆ® Ç¥Çö½Ä¿¡¼­ []¸¦ ()·Î ¹Ù²Û °Í Ã³·³ (½Ä for º¯¼ö in ¹Ýº¹°¡´ÉÇÑ°´Ã¼ -Á¶°Ç½Ä-)·Î Á¦³Ê·¹ÀÌÅÍ Ç¥Çö½ÄÀ» ¸¸µé ¼ö ÀÖÀ½.

## coroutine
***
- ÄÚ·çÆ¾: Æ¯Á¤ ½ÃÁ¡¿¡ »ó´ë¹æÀÇ ÄÚµå¸¦ ½ÇÇà. ¸ÞÀÎ ·çÆ¾°ú ÄÚ·çÆ¾ÀÇ ÄÚµå¸¦ ¹ø°¥¾Æ°¡¸ç ½ÇÇàÇÑ´Ù.
- yield : Æò¹üÇÑ ÇÔ¼ö¿¡ ¹«ÇÑ ¹Ýº¹¹®°ú yield¸¦ »ç¿ëÇØ ÄÚ·çÆ¾(Á¦³×·¹ÀÌÅÍ ±â¹Ý)À¸·Î ¸¸µé ¼ö ÀÖ°í, ¾Æ·¡ÀÇ ÃÖÃÊ ½ÇÇà½Ã yield±îÁö ÇÔ¼ö¸¦ ½ÇÇà½ÃÅ²´Ù.  
- ÄÚ·çÆ¾À» °è¼Ó À¯Áö½ÃÅ°±â À§ÇØ ¹«ÇÑ ·çÇÁ¸¦ »ç¿ëÇÏ°í, next()³ª send(NONE)·Î ÄÚ·çÆ¾ °´Ã¼¸¦ ÃÖÃÊ ½ÇÇàÇÏ¿© ÄÚ·çÆ¾ÀÇ µ¿ÀÛÀ» ½ÇÇàÇÑ´Ù.
- ÄÚ·çÆ¾.send(°ª) À¸·Î ÄÚ·çÆ¾¿¡ °ªÀ» º¸³»¸ç ÄÚµå¸¦ ½ÇÇàÇÒ ¼ö ÀÖ°í, °ª=(yield) ·Î send ¸Þ¼­µå°¡ º¸³½ °ªÀ» ¹Þ¾Æ ¿Ã ¼ö ÀÖ´Ù.
- ÄÚ·çÆ¾¿¡¼­ º¯¼ö = (yield º¯¼ö)·Î °ªÀ» ¹Þ¾Æ ¿À¸é¼­ °ªÀ» ¹ÛÀ¸·Î º¸³¾ ¼ö ÀÖ°í, º¯¼ö = ÄÚ·çÆ¾°´Ã¼.send(°ª),º¯¼ö=next(ÄÚ·çÆ¾)À¸·Î yield ¿¡¼­ º¸³½ °ªÀ» ¹Þ¾Æ ¿Ã ¼ö ÀÖ´Ù.
- ÄÚ·çÆ¾.close ·Î ÄÚ·çÆ¾À» Á¾·áÇÒ ¼ö ÀÖ´Ù. ÀÌ¶§´Â GeneratorExit ¿¹¿Ü°¡ ¹ß»ýÇÑ´Ù.
- ÄÚ·çÆ¾.throw(¿¹¿ÜÀÌ¸§, ¿¡·¯¸Þ¼¼Áö)·Î ÄÚ·çÆ¾¿¡ ¿¹¿Ü¸¦ ¹ß»ý½ÃÅ³ ¼ö ÀÖ°í, ÄÚ·çÆ¾¿¡ except ¸¦ »ç¿ëÇÒ ¼ö ÀÖ´Âµ¥, ±× ¾È¿¡¼­ yield ¸¦ »ç¿ëÇÏ¸é throw ÀÇ ¹ÝÈ¯°ªÀ¸·Î ³ª¿Â´Ù.
- º¯¼ö = yield from ÄÚ·çÆ¾()À¸·Î ÄÚ·çÆ¾ ³»¿¡¼­ÀÇ return ¹ÝÈ¯°ªÀ» ¹Þ¾Æ¿Ã ¼ö ÀÖ´Ù.
- ÀÌ¶§ º¯¼ö°¡ ÀÖ´Â »óÀ§ ÄÚ·çÆ¾¿¡ °ªÀ» º¸³»µµ ÇÏÀ§ ÄÚ·çÆ¾¿¡¼­ °ªÀ» ¹Þ°í, ÇÏÀ§ ÄÚ·çÆ¾¿¡¼­ yield ·Î °ªÀ» º¸³»¸é ±×°É ±×´ë·Î ´Ù½Ã ¹ÛÀ¸·Î º¸³¿.
### asyncio
- asyncio : ºñµ¿±â ÇÁ·Î±×·¡¹ÖÀ» À§ÇÑ ¸ðµâ. CPUÀÛ¾÷°ú I/O¸¦ º´·Ä·Î Ã³¸®ÇÏ°Ô ÇØÁÜ.
- ³×ÀÌÆ¼ºê ÄÚ·çÆ¾ : Á¦³Ê·¹ÀÌÅÍ±â¹Ý ÄÚ·çÆ¾(yield)°ú ´Þ¸® async·Î ¸¸µç ÄÚ·çÆ¾.  
- [async def ÇÔ¼ö¸í():] : ºñµ¿±â(async)ÇÔ¼ö ¼±¾ð(³×ÀÌÆ¼ºê ÄÚ·çÆ¾ ¼±¾ð)
  
- loop = asyncio.get_event_loop()  : ÀÌº¥Æ® ·çÇÁ¸¦ ¾òÀ½.
- loop.run_until_complete(asyncÇÔ¼ö()) : ÇÔ¼ö°¡ ³¡³¯ ¶§±îÁö ±â´Ù¸².
- loop.close() : ·çÇÁ Á¾·á.
  
- [await ÄÚ·çÆ¾/Ç»Ã³/ÅÂ½ºÅ© °´Ã¼] : ÇØ´ç °´Ã¼°¡ ³¡³¯¶§±îÁö ±â´Ù¸° µÚ °á°ú¸¦ ¹ÝÈ¯. ³×ÀÌÆ¼ºê ÄÚ·çÆ¾ ³»ºÎ¿¡¼­¸¸ »ç¿ë°¡´É. ÄÚ·çÆ¾ ¾È¿¡¼­ ´Ù¸¥ ÄÚ·çÆ¾ ½ÇÇà½Ã »ç¿ë.
- Ç»Ã³/ÅÂ½ºÅ© : asyncio.Future- ¹Ì·¡¿¡ ÇÒÀÏÀ» Ç¥ÇöÇÏ´Â Å¬·¡½º. ÇÒÀÏ Ãë¼Ò/»óÅÂÈ®ÀÎ/¿Ï·á/°á°ú¼³Á¤¿¡ »ç¿ë. | asyncio.Task- Ç»Ã³ÀÇ ÆÄ»ýÅ¬·¡½º. Ç»Ã³ÀÇ ±â´É°ú ½ÇÇàÇÒ ÄÚ·çÆ¾°´Ã¼¸¦ Æ÷ÇÔ.
- [await asyncio.sleep(i)] : ÇÔ¼ö ³»ºÎ¿¡¼­ iÃÊ¸¸Å­ sleep. ³×ÀÌÆ¼ºê ÄÚ·çÆ¾ÀÌ±â¿¡ await°ú ÇÔ²¾ »ç¿ëÇØÁà¾ß ÇÔ.
- asyncio.run(asyncÇÔ¼ö()) : ÇÔ¼ö ½ÇÇà.

# decorator
***
- µ¥ÄÚ·¹ÀÌÅÍ:@·Î ½ÃÀÛ. ¸Þ¼­µå¸¦ Àå½Ä. Àå½ÄÀÚ¶ó°íµµ ÇÔ. ÇÔ¼ö¸¦ ¼öÁ¤ÇÏÁö ¾ÊÀº »óÅÂ¿¡¼­ Ãß°¡ ±â´ÉÀ» ±¸ÇöÇÏ°í ½ÍÀ» ¶§ »ç¿ë.
- È£ÃâÇÒ ÇÔ¼ö¸¦ ¸Å°³º¯¼ö·Î ¹ÞÀº ÈÄ Ãß°¡ÇÒ ±â´ÉÀ» ´ãÀº ÇÔ¼ö¸¦ ¸¸µé°í, ±×°ÍÀ» ¹ÝÈ¯ÇÏ´Â ½ÄÀ¸·Î Á¦ÀÛ°¡´É.
- ÇÔ¼ö À§¿¡ @µ¥ÄÚ·¹ÀÌÅÍ¸¦ »ç¿ëÇØ ÇÔ¼ö È£Ãâ½Ã¸¶´Ù µ¥ÄÚ·¹ÀÌÅÍ »ç¿ë. µ¥ÄÚ·¹ÀÌÅÍ¸¦ ¿©·¯°³ »ç¿ëÇÏ´Â °Íµµ °¡´ÉÇÏ¸ç, ÀÌ ¶§´Â À§¿¡¼­ ºÎÅÍ ½ÇÇàÇÑ´Ù.

## make deco
***
- ¸Å°³º¯¼ö¿Í ¹ÝÈ¯°ªÀÌ ÀÖ´Â ÇÔ¼öÀÇ µ¥ÄÚ·¹ÀÌÅÍ´Â, ¹Ù±ùÀÇ ÇÔ¼ö´Â ¿©ÀüÈ÷ ÇÔ¼ö¸¦ ÀÎ¼ö·Î ¹Þ¾Æ ³»ºÎ ÇÔ¼ö¸¦ ¹ÝÈ¯ÇÏ°Ô ÇÏ°í, µ¥ÄÚ·¹ÀÌÅÍ ³»ºÎÀÇ ÇÔ¼ö¿¡¼­ 
  È£ÃâÇÒ ÇÔ¼öÀÇ ¸Å°³º¯¼ö¿Í °°Àº ¸Å°³º¯¼ö¸¦ ÁöÁ¤ÇÏ°í, ÇÔ¼ö¿¡ º¯¼öµéÀ» ³Ö¾î È£ÃâÇÑ ÈÄ, ±× ¹ÝÈ¯°ªÀ» º¯¼ö¿¡ ÀúÀåÇÏ°í, ±×°ÍÀ» ¸®ÅÏÇÏ°Ô ³»ºÎÇÔ¼ö¸¦ ¸¸µé¸é µÈ´Ù.
- ¸Å°³º¯¼öÀÇ °³¼ö°¡ °íÁ¤µÇ¾îÀÖÁö ¾Ê´Ù¸é °¡º¯ÀÎ¼öÇÔ¼ö·Î ¸¸µé¾îÁÖ¸é µÈ´Ù. ¸Å°³º¯¼ö¸¦(*args, **kwargs)½ÄÀ¸·Î ¹ÞÀº ÈÄ, °°Àº½ÄÀ¸·Î ¾ðÆÐÅ·ÇØ¼­ ÇÔ¼ö¿¡ ³Ö¾îÁÖ¸é µÈ´Ù. 
  ÀÌ¶§, À§Ä¡ÀÎ¼ö¿Í Å°¿öµå ÀÎ¼ö¸¦ ¸ðµÎ ¹ÞÀ» ¼ö ÀÖµµ·Ï µÑ ´Ù ÁöÁ¤ÇØÁØ´Ù.
- Å¬·¡½º ¾ÈÀÇ ¸Þ¼­µå¿¡ µ¥ÄÚ·¹ÀÌÅÍ¸¦ »ç¿ëÇÏ·Á¸é ³»ºÎ ÇÔ¼öÀÇ Ã¹¹øÂ° ¸Å°³º¯¼ö¸¦ self(Å¬·¡½º ¸Þ¼­µå´Â cls)·Î ÇØÁÖ¾î¾ß ÇÏ¸ç, ³»ºÎÇÔ¼ö¿¡¼­ func¸¦ 
  È£ÃâÇÒ ¶§¿¡µµ self ¸¦ Ã¹¹øÂ° ¸Å°³º¯¼ö·Î ³Ö¾îÁà¾ß ÇÑ´Ù.

- µ¥ÄÚ·¹ÀÌÅÍ¿¡¼­ ¸Å°³º¯¼ö¸¦ »ç¿ëÇÏ·Á¸é °¡Àå ¹Ù±ùÀ» ¸Å°³º¯¼ö¸¦ ¹Þ´Â ÇÔ¼ö·Î ½ÎÁØ´ÙÀ½ °¡Àå ¹Ù±ù¿¡¼­ µ¥ÄÚ·¹ÀÌÅÍ¸¦ ¹ÝÈ¯ÇÏ°Ô ÇÏ¸ç, »ç¿ëÀº @µ¥ÄÚ·¹ÀÌÅÍ(ÀÎ¼ö)·Î ÇÑ´Ù.
- µ¥ÄÚ·¹ÀÌÅÍ¸¦ ¿©·¯°³ »ç¿ëÇÏ¸é ±× ÇÔ¼öÀÇ ÀÌ¸§ÀÌ ¾Æ´Ñ ³»ºÎ ÇÔ¼öÀÇ ÀÌ¸§ÀÌ ³ª¿Ã ¼ö ÀÖ´Âµ¥, ÀÌ¶§´Â functools ¸ðµâÀÇ wraps µ¥ÄÚ·¹ÀÌÅÍ¸¦ »ç¿ëÇØ  
  @functools.wraps(func)½ÄÀ¸·Î ³»ºÎÇÔ¼ö À§¿¡ ÁöÁ¤ÇØÁØ´Ù. ¿ø·¡ ÇÔ¼öÀÇ Á¤º¸¸¦ À¯Áö½ÃÄÑÁÖ¾î À¯¿ëÇÏ´Ù.

## class deco
***
- Å¬·¡½º·Î µ¥ÄÚ·¹ÀÌÅÍ¸¦ ¸¸µé¶§´Â ¸ÕÀú È£ÃâÇÒ ÇÔ¼ö¸¦ ÃÊ±ê°ªÀ¸·Î ¹Þ¾Æ ¼Ó¼ºÀ¸·Î ÀúÀåÇÑ µÚ, __call__ÇÔ¼ö¸¦ ¸¸µé¾î ÇÔ¼öÀÇ ¾Õ µÚ·Î ½ÇÇàÇÒ ¸í·ÉÀ» ÁöÁ¤ÇÑ µÚ
  »çÀÌ¿¡ self.func()·Î ¹ÞÀº ÇÔ¼ö¸¦ ½ÇÇàÇÏ°Ô ÇÑ´Ù.
- »ç¿ë½Ã¿¡´Â ´Ù¸¥ µ¥ÄÚ·¹ÀÌÅÍ¿Í ¸¶Âù°¡Áö·Î @·Î »ç¿ëÇÏ°Å³ª, µ¥ÄÚ·¹ÀÌÅÍ(ÇÔ¼ö)·Î ÀÎ½ºÅÏ½º »ý¼º ÈÄ ÀÎ½ºÅÏ½º¸¦ È£ÃâÇÏ¸é µÈ´Ù.
- Å¬·¡½º·Î ¸Å°³º¯¼ö¿Í ¹ÝÈ¯°ªÀ» Ã³¸®ÇÒ ¶§µµ, __call__ÇÔ¼ö¿¡ self ¿Í ½ÇÇàÇÒ Å¬·¡½ºÀÇ ¸Å°³º¯¼ö(*args,**kwargs)¸¦ ¸Å°³º¯¼ö·Î ¹Þ°í,
  ÇÔ¼ö ½ÇÇà½Ã¿¡ ¸Å°³º¯¼ö¸¦ ³Ö°í, ¹ÝÈ¯°ªÀ» µû·Î ¹Þ¾Æ ±×°ÍÀ» ¹ÝÈ¯ÇÏ°Ô ÇÏ¸é µÈ´Ù.
- ¸Å°³º¯¼ö°¡ ÀÖ´Â µ¥ÄÚ·¹ÀÌÅÍ´Â __init__¿¡¼­ »ç¿ëÇÒ ¸Å°³º¯¼ö¸¦ ¹Þ°í, __call__¿¡¼­ func ¸¦ ¸Å°³º¯¼ö·Î ¹Þ°í,
  ±× ¾È¿¡ wrapper(a,b)°°ÀÌ »ç¿ëÇÒ ¸Å°³º¯¼ö°¡ ÀÖ´Â ÇÔ¼ö¸¦ »ý¼ºÇÑ´Ù.

- def type_check(type_a, type_b): > µ¥ÄÚ·¹ÀÌÅÍ°¡ ¹ÞÀ» ¸Å°³º¯¼ö
-  def real_decorator(func): > ½ÇÁ¦ µ¥ÄÚ·¹ÀÌÅÍ. °¨½Ò ÇÔ¼ö¸¦ ÀÎ¼ö·Î ¹ÞÀ½
-    def wrapper(a, b): > ¾î¶»°Ô Àå½ÄÇÒÁö¸¦ ÀÛ¼º, °¨½Ò ÇÔ¼öÀÇ ÀÎ¼ö¸¦ ¸Å°³º¯¼ö·Î ¹ÞÀ½.
     È¤½Ã func ¿¡ ¸®ÅÏÀÌ ÀÖ´Ù¸é, rÀ» ¹ÝÈ¯½ÃÅ³ ¶§ ¾ÕµÚ·Î °¨½Ò ¹®ÀÚ¿­µµ °°ÀÌ ¹ÝÈ¯ÇØ¾ß ¹ÝÈ¯°ª¸¸ µÚ·Î ¹Ð·Á³ªÁö ¾ÊÀ½.
- ÀÌ·±½ÄÀ¸·Î µ¥ÄÚ·¹ÀÌÅÍ¸¦ ÀÛ¼º.

# regular expression
***
- Á¤±Ô Ç¥Çö½Ä : ÀÏÁ¤ÇÑ ±ÔÄ¢À» °¡Áø ¹®ÀÚ¿­À» Ç¥ÇöÇÏ´Â ¹æ¹ý.
- ¹®ÀÚ¿­ ÆÇ´Ü : re ¸ðµâÀÇ match ÇÔ¼ö(¹®ÀÚ¿­ Ã³À½ºÎÅÍ ¸ÅÄ¡µÇ´ÂÁö ÆÇ´Ü)¿¡ re.match('ÆÐÅÏ', 'ÆÇ´ÜÇÒ ¹®ÀÚ¿­')½ÄÀ¸·Î ³Ö¾î ÆÐÅÏ(ÀÖ´ÂÁö °Ë»çÇÒ ¹®ÀÚ)ÀÌ ÀÖÀ¸¸é ¸ÅÄ¡°´Ã¼°¡ ¹ÝÈ¯µÇ°í, 
  ¾øÀ¸¸é ¾Æ¹«°Íµµ ¹ÝÈ¯ÇÏÁö ¾ÊÀ½. ¹®ÀÚ¿­.find("¹®ÀÚ¿­")°ú °°Àº ±â´É.
- ¹®ÀÚ¿­ ÆÇ´Ü (ÇÏ³ª¶óµµ) : re.match('hello|world', 'hello') Ã³·³ ÁöÁ¤µÈ ¹®ÀÚ¿­ÀÌ ÇÏ³ª¶óµµ Æ÷ÇÔµÇ´ÂÁö ÆÇ´ÜÇÔ.
- ¹®ÀÚ¿­ ÆÇ´Ü (À§Ä¡) :  re.search('ÆÐÅÏ', '¹®ÀÚ¿­')Ã³·³ search ÇÔ¼ö(¹®ÀÚ¿­ ÀÏºÎºÐ¿¡¼­ ¸ÅÄ¡µÇ´ÂÁö ÆÇ´Ü)¿¡ "^¹®ÀÚ¿­"·Î ¸Ç ¾Õ¿¡ ¿À´ÂÁö, 
  "¹®ÀÚ¿­$"·Î ¸Ç µÚ¿¡ ¿À´ÂÁö ÆÇ´ÜÇÔ. ±× ¹üÀ§·Î ½ÃÀÛÇÏ´ÂÁö º¸´Â °Ç ^[¹üÀ§\]\*/+, ³¡³ª´ÂÁö º¸´Â°Ç [¹üÀ§\]*/+$ÀÌ´Ù.
- ¹®ÀÚ¿­ ÆÇ´Ü (¼ýÀÚ·Î?) : re.match('\[0-9]*', '1234') Ã³·³ ¼ýÀÚÀÇ ¹üÀ§´Â []¿¡ 0-9½ÄÀ¸·Î Ç¥ÇöÇÏ¸ç, *´Â 0°³ÀÌ»ó. +´Â 1°³ÀÌ»óÀÎÁö ÆÇ´ÜÇÑ´Ù.
- ¹®ÀÚ¿­ ÆÇ´Ü ±âÈ£(+,\*) : a+b ÀÏ¶§´Â aµµ bµµ ÇÏ³ª ÀÌ»ó ÀÖ¾î¾ß ÇÏ°í, a*bÀÏ ¶§´Â b¸¸ ÀÖ¾îµµ ¸ÅÄªµÈ´Ù.
- ¹®ÀÚ¿­ ÆÇ´Ü ±âÈ£(ÇÏ³ª¸¸) : ¹®ÀÚ? ³ª ¹üÀ§? ´Â ±× ¹®ÀÚ³ª ¹üÀ§°¡ 0°³ ¶Ç´Â ÇÏ³ªÀÎÁö ÆÇ´ÜÇÏ°í('abc?d'½Ä), .Àº .ÀÇ À§Ä¡¿¡ ¾Æ¹« ¹®ÀÚ³ª ¼ýÀÚ°¡ 1°³ ÀÖ´ÂÁö ÆÇ´ÜÇÑ´Ù("ab.d"½Ä).
- ¹®ÀÚ¿­ ÆÇ´Ü °³¼ö : ¹®ÀÚ{°³¼ö} ¿Í (¹®ÀÚ¿­){°³¼ö}, \[¹üÀ§]{°³¼ö}·Î ±× ¹®ÀÚ(¿­)ÀÌ °³¼ö¸¸Å­ ÀÖ´ÂÁö ÆÇ´ÜÇÑ´Ù.
- ¹®ÀÚ¿­ ÆÇ´Ü °³¼ö(¹üÀ§) : ¹®ÀÚ{½ÃÀÛ °³¼ö,³¡ °³¼ö}·Î ±× °³¼ö ¹üÀ§ ¾È¿¡ ¹®ÀÚ°³¼ö°¡ µé¾î°¡´ÂÁö ÆÇ´ÜÇÔ.
- ¹®ÀÚ¿­ ÆÇ´Ü (È¥ÇÕ ¹üÀ§) : \[A-Z0-9] Ã³·³ ¹üÀ§ µÎ°³¸¦ ¼­·Î ºÙ¿©¼­ Ç¥Çö. ÀÌ °æ¿ì´Â ´ë¹®ÀÚ,¼ýÀÚ°¡ ¾øÀÌ ¼Ò¹®ÀÚ¸¸ ÀÖ´Ù¸é ¸ÅÄªµÇÁö ¾ÊÀ½. 
  ÇÑ±ÛÀÇ °æ¿ìµµ °¡-ÆR Ã³·³ ³ª¿Ã ¼ö ÀÖ´Â ÇÑ±Û Á¶ÇÕÀ» Á¤ÇØÁÖ¸é µÈ´Ù.
- ¹®ÀÚ¿­ ÆÇ´Ü (È¥ÇÕ ¹üÀ§ ¹þ¾î³²) : \[^A-Z0-9] Ã³·³ ¹üÀ§ ¾Õ¿¡ ^¸¦ ºÙÀÌ¸é ÇØ´çÇÏ´Â ¹üÀ§¸¦ ¹þ¾î³ª´ÂÁö ÆÇ´ÜÇÔ. Æ÷ÇÔµÇÁö ¾Ê¾Æ¾ß ¸ÅÄª.
- ¹®ÀÚ¿­ ÆÇ´Ü (Æ¯¼ö¹®ÀÚ) : \Æ¯¼ö¹®ÀÚ ·Î ±× Æ¯¼ö¹®ÀÚ°¡ ¹üÀ§¿¡ µé¾î°¡´ÂÁö ÆÇ´ÜÇÒ ¼ö ÀÖ´Ù. Æ¯¼ö¹®ÀÚ°¡ ¹üÀ§([])¿¡ µé¾î°¡ ÀÖÀ¸¸é ºÙÀÌÁö ¾Ê¾Æµµ µÇÁö¸¸, ¿À·ù°¡ ³­´Ù¸é ºÙÀÌ¸é µÈ´Ù. 
  '[$()a-zA-Z0-9\]+'°°Àº½Ä.
- ¹®ÀÚ¿­ ÆÇ´Ü (´Ü¼ø ¼ýÀÚ,¹®ÀÚ) : \d > ¸ðµç ¼ýÀÚ, \D > ¼ýÀÚ Á¦¿Ü ¸ðµç ¹®ÀÚ, \w > ¿µ¹®´ë¼Ò¹®ÀÚ+¼ýÀÚ+¹ØÁÙ, \W > \wÁ¦¿Ü ¸ðµç ¹®ÀÚ(ÇÑ±ÛÀÌ³ª ´Ù¸¥ Æ¯¼ö¹®ÀÚµî). 
  ÀÌ·±½ÄÀ¸·Î °£´ÜÇÏ°Ô Ç¥ÇöÇÒ ¼ö ÀÖ´Ù. '\d+'½ÄÀ¸·Î »ç¿ë.
- ¹®ÀÚ¿­ ÆÇ´Ü (°ø¹é) : " "·Î Ã³¸®ÇØµµ µÇ°í, \s: [ \t\n\r\f\v\], \S: [^ \t\n\r\f\v\](°ø¹é Á¦¿Ü ´Ù¸¥ ¹®ÀÚ¸¸ Æ÷ÇÔ) À¸·Î °£´ÜÇÏ°Ô »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- °°Àº Á¤±ÔÇ¥Çö½ÄÀ» ÀÚÁÖ »ç¿ëÇÑ´Ù¸é °´Ã¼ = re.compile('ÆÐÅÏ'), °´Ã¼.match/search('¹®ÀÚ¿­') ·Î ¸Þ¼­µå¸¦ È£ÃâÇÏ´Â°Ô ´õ È¿À²ÀûÀÌ´Ù.

## grob
***
- ±×·ì :  ÆÐÅÏ¿¡ ³Ö´Â ¹®ÀÚ¿­À» (Á¤±ÔÇ¥Çö½Ä) (Á¤±ÔÇ¥Çö½Ä)Ã³·³ ¹­¾î¼­ ¸¸µê.
- ¸ÅÄ¡°´Ã¼.group(±×·ì¼ýÀÚ) ·Î ÇØ´ç ±×·ì¿¡ ¸ÅÄªµÈ ¹®ÀÚ¿­À» °¡Á®¿Ã ¼ö ÀÖÀ½. ¼ýÀÚ¿¡ 0À» ³Ö°Å³ª ³ÖÁö ¾ÊÀ¸¸é ¸ðµç ¹®ÀÚ¿­À» ÇÑ²¨¹ø¿¡ ¹ÝÈ¯ÇÑ´Ù.
- (?P<ÀÌ¸§>Á¤±ÔÇ¥Çö½Ä)À¸·Î ±×·ì¿¡ ÀÌ¸§À» ÁöÁ¤ÇÒ ¼ö ÀÖ°í, ¸ÅÄ¡°´Ã¼.group('±×·ìÀÌ¸§')À¸·Î »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- re.findall('ÆÐÅÏ', '¹®ÀÚ¿­')À¸·Î ÆÐÅÏ¿¡ ¸ÅÄªµÇ´Â ¸ðµç ¹®ÀÚ¿­À» ¸®½ºÆ®ÀÇ ÇüÅÂ·Î °¡Á®¿Ã ¼ö ÀÖ´Ù.
- (.[a-z\]+)*´Â Á¡°ú ¿µ¹® ¼Ò¹®ÀÚ°¡ 1°³ ÀÌ»ó ÀÖ´ÂÁö ÆÇ´ÜÇÏ°í, ÀÌ°Í ÀÚÃ¼°¡ 0°³ ÀÌ»óÀÎÁö ÆÇ´ÜÇÔ. Áï,  ¹Ýµå½Ã ÁöÄÑ¾ß ÇÏÁö¸¸ ÀÖ¾îµµ µÇ°í ¾ø¾îµµ µÇ´Â »óÈ²¿¡ »ç¿ë.

## sub
***
- re.sub('ÆÐÅÏ', '¹Ù²Ü¹®ÀÚ¿­', '¹®ÀÚ¿­', ¹Ù²ÜÈ½¼ö)·Î ¹®ÀÚ¿­À» ¹Ù²Ü ¼ö ÀÖ°í, È½¼ö¸¦ »ý·«ÇÏ¸é ¸ðµç ¹®ÀÚ¿­À» ¹Ù²Û´Ù.
- re.sub('ÆÐÅÏ', ±³Ã¼ÇÔ¼ö(¸ÅÄ¡°´Ã¼¸¦ ¸Å°³º¯¼ö·Î ¹Þ¾Æ ¹Ù²Ü ¹®ÀÚ¿­À» ¹ÝÈ¯. ¶÷´ÙÇ¥Çö½ÄÀ» »ç¿ëÇÏ´Â °Íµµ °¡´ÉÇÔ.), '¹®ÀÚ¿­', ¹Ù²ÜÈ½¼ö)·Îµµ »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- ÆÐÅÏ¿¡¼­ ±×·ìÀ» ¹­Àº µÚ ¹Ù²Ü ¹®ÀÚ¿­¿¡¼­ '\\2 \\1 \\2 \\1'Çü½ÄÀ¸·Î ÁöÁ¤ÇØÁÖ¸é, ±× ±×·ì¿¡ ¸ÅÄ¡µÈ ¹®ÀÚ¿­À» ±× À§Ä¡¿¡ »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- ±×·ì¿¡ ÀÌ¸§À» Áö¾ú´Ù¸é, \\g<ÀÌ¸§>Çü½ÄÀ¸·Îµµ ¸ÅÄªµÈ ¹®ÀÚ¿­À» ÁöÁ¤ÇÒ ¼ö ÀÖ´Ù.
- ¹®ÀÚ¿­ ¾Õ¿¡ rÀ» ºÙÀÌ¸é ¿ø½Ã ¹®ÀÚ¿­ÀÌ µÇ¾î \¸¦ ºÙÀÌÁö ¾Ê¾Æµµ Æ¯¼ö¹®ÀÚ¸¦ ÆÇ´ÜÇÒ ¼ö ÀÖ°ÔµÇ¾î r'\¼ýÀÚ \g<ÀÌ¸§> \g<¼ýÀÚ>'Ã³·³ \¸¦ ÇÏ³ª¸¸ ºÙ¿©¼­ »ç¿ëÇÒ ¼ö ÀÖ°Ô µÈ´Ù.

# pip
***
- pip > ¿øµµ¿ì¿ë ÆÄÀÌ½ã¿¡´Â ±âº» ³»Àå, ¸®´ª½º : ¸®´ª½º, macOS$ curl -O https://bootstrap.pypa.io/get-pip.py(curl ¼³Ä¡ ÇÊ¿ä), ¸ÆOS : 
  $ sudo python3 get-pip.py ·Î ¼³Ä¡ÇÒ ¼ö ÀÖ´Ù.
- (Window ±âÁØ) pip install ÆÐÅ°Áö¸í À¸·Î ÆÐÅ°Áö¸¦ ¼³Ä¡ÇÒ ¼ö ÀÖ°í, -m(¸ðµâ½ÇÇà ¿É¼Ç)À» python °ú pip »çÀÌ¿¡ ³Ö¾î ½ÇÇàÇÒ ¼öµµ ÀÖ´Ù.
- pip search ÆÐÅ°Áö: ÆÐÅ°Áö °Ë»ö
- pip install ÆÐÅ°Áö==¹öÀü: Æ¯Á¤ ¹öÀüÀÇ ÆÐÅ°Áö¸¦ ¼³Ä¡(¿¹: pip install requests==2.9.0)
- pip list ¶Ç´Â pip freeze: ÆÐÅ°Áö ¸ñ·Ï Ãâ·Â
- pip uninstall ÆÐÅ°Áö: ÆÐÅ°Áö »èÁ¦

- ¸ðµâ(module): º¯¼ö, ÇÔ¼ö, Å¬·¡½º µîÀ» ¸ð¾Æ ³õÀº ½ºÅ©¸³Æ® ÆÄÀÏ.
- ÆÐÅ°Áö(package): ¿©·¯ ¸ðµâÀ» ¹­Àº °Í

# import
***
- import ¸ðµâ as º°¸í > ¸ðµâ ÀÌ¸§´ë½Å º°¸íÀ¸·Îµµ ±¸µ¿ °¡´É.
- from ¸ðµâ import º¯¼ö,ÇÔ¼ö,Å¬·¡½º > ¸ðµâ.º¯¼ö·Î ¾µ ÇÊ¿ä ¾øÀÌ ±×³É º¯¼ö¸¸ ¾µ ¼ö ÀÖ´Ù. ±× ÇÔ¼ö¸¸ »ç¿ëÇÒ ¶§ À¯¿ë. *·Î ¾²¸é ¸ðµç º¯¼ö,ÇÔ¼ö,Å¬·¡½º¸¦ °¡Á®¿Â´Ù. 
  Å¬·¡½ºÀÇ °æ¿ì ¸ðµâ.Å¬·¡½º·Î »ç¿ëÇØ¾ß ÇÏ´Ï ÀÌ ±â´ÉÀ» ¾²¸é ÁÁ´Ù.
- from ¸ðµâ import º¯¼ö as º¯¸í > °¡Á®¿Â º¯¼ö,ÇÔ¼ö,Å¬·¡½º¿¡ º°¸íÀ» ºÙÀÎ´Ù.
- from ¸ðµâ import º¯¼ö as ÀÌ¸§1, ÇÔ¼ö as ÀÌ¸§2, Å¬·¡½º as ÀÌ¸§3 > ¿©·¯°³¸¦ °¡Á®¿À¸ç °¢°¢¿¡ ÀÌ¸§À» ºÙÀÎ´Ù.
- import ¸¦ ÇØÁ¦ÇÏ·Á¸é del ¸ðµâ ·Î ÇÒ ¼ö ÀÖ°í, ´Ù½Ã °¡Á®¿À·Á¸é importlib ¸ðµâÀÇ reload ¸¦ »ç¿ëÇÏ¸é µÈ´Ù.
- import ÆÐÅ°Áö.¸ðµâ, import ÆÐÅ°Áö.¸ðµâ1, ÆÐÅ°Áö.¸ðµâ2 ·Î ÆÐÅ°Áö¿Í ¸ðµâÀ» °¡Á®¿Ã ¼ö ÀÖ°í ¿©±â¿¡ as¸¦ »ç¿ëÇÏ¸é ÆÐÅ°Áö.¸ðµâ ´ë½Å º°¸í¸¸ »ç¿ëÇÒ ¼ö ÀÖ°í 
  from µµ from ÆÐÅ°Áö.¸ðµâ import º¯¼ö ½ÄÀ¸·Î »ç¿ëÇÑ´Ù..

- °°Àº Æú´õ¿¡ ÀÖ´Â ÆÄÀÏ(¸ðµâ)Àº ±×³É import ¸ðµâ·Î »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- __name__Àº Á÷Á¢ ½ÇÇàÇÑ ÆÄÀÏÀÏ °æ¿ì __main__ÀÌ ¹ÝÈ¯µÇ°í, ¾Æ´Ï¸é ±× ÆÄÀÏ(¸ðµâ)ÀÇ ÀÌ¸§ÀÌ ¹ÝÈ¯µÈ´Ù. ÆÐÅ°ÁöÀÇ ¸ðµâÀÏ °æ¿ì ÆÐÅ°Áö.¸ðµâÀÌ¸§ ÀÌ ¹ÝÈ¯µÈ´Ù.

## about import
***
- \_\_all__ = ["ÆÐÅ°Áö","ÀÇ ¿ä¼Òµé"\] ·Î *À» ÅëÇØ ¸ðµç ¿ä¼Ò¸¦ ºÒ·¯¿Ã¶§ °ø°³ÇÒ °ÍµéÀÇ ¸ñ·ÏÀ» ÁöÁ¤ÇÒ ¼ö ÀÖ´Ù.
- ÇöÀç Æú´õ¿¡ ¸ðµâ, ÆÐÅ°Áö°¡ ¾øÀ¸¸é path ¿¡¼­ Ã£´Âµ¥, ÀÌ°Ç sys ¸ðµâÀÇ path ÇÔ¼ö·Î º¼ ¼ö ÀÖ´Ù. ±× ¾ÈÀÇ site-packages ´Â pip ·Î ¼³Ä¡ÇÑ ÆÐÅ°Áö°¡ µé¾î°¡´Âµ¥, 
  °¡»óÈ¯°æÀ» ¸¸µç´Ù¸é °¡»óÈ¯°æ/Lib/site-packages Æú´õ¿¡ µé¾î°£´Ù.
- ÆÄÀÌ½ãÀº ÇÏÀ§ ÆÐÅ°Áö¸¦ ¸¸µé ¼ö ÀÖ°í, ÇÏÀ§ÆÐÅ°ÁöÀÇ ¸ðµâÀ» °¡Á® ¿Ã ‹š´Â °èÃþ¼ø¼­´ë·Î .À» ºÙ¿©(.operation.element)°¡Á®¿À¸é µÈ´Ù. 
  ÇÏÀ§ÆÐÅ°Áö¿¡¼­ ¿·ÀÇ ÆÐÅ°Áö¸¦ °¡Á® ¿Ã ¶§´Â ..(»óÀ§Æú´õ¶õ ¶æ, À§·Î ¿Ã¶ó°¥¼ö·Ï .ÀÌ ´Ã¾î³²)ÆÐÅ°Áö ·Î ºÒ·¯¿Ã ¼ö ÀÖ´Ù.
- ÆÄÀÌ½ã¿¡¼­ \_\_init__.py ÆÄÀÏÀº Æú´õ°¡ ÆÐÅ°Áö·Î ÀÎ½ÄµÇ°Ô ÇÏ±âµµ ÇÏ°í, ÆÐÅ°Áö¸¦ ÃÊ±âÈ­ÇÏ´Â ¿ªÇÒµµ ÇÑ´Ù. Áï import ½Ã¿¡ ÀÌ ÆÄÀÏÀÌ ½ÇÇàµÈ´Ù. 
  __all__µµ ÀÌ ÆÄÀÏ¿¡¼­ ÇØ¾ß ÇÏ°í, ºñ¿öµÑ¼öµµ ÀÖÀ¸¸ç, 3.3ÀÌ»ó¿¡¼± ¾ø¾îµµ ÆÐÅ°Áö·Î ÀÎ½ÄµÈ´Ù. ±Ùµ¥ ÀÌ°Ô ºñ¾îÀÖÀ¸¸é from ÀÌ ¾ÈµÇ´Â µí.
- ÀÌ¸¦ ÀÌ¿ëÇØ __init__ÆÄÀÏ¿¡ from .(ÇöÀçÆÐÅ°Áö) import ¸ðµâ  ½ÄÀ¸·Î ¸í·ÉÀ» ³Ö¾î ÆÐÅ°Áö¸¸ °¡Á®¿Íµµ ¸ðµâµµ °°ÀÌ °¡Á®¿Í calcpkg.operation.add(10, 20)½ÄÀ¸·Î 
  »ç¿ëÇÒ ¼ö ÀÖ°Ô ¸¸µé ¼ö ÀÖ´Ù.
- ÆÐÅ°Áö¸¦ import ÇÒ¶§ from .¸ðµâ import º¯¼ö, ÇÔ¼ö, Å¬·¡½º ¸¦ »ç¿ëÇÏ¸é ¾Õ¿¡ ¸ðµâÀ» ºÙÀÏ ÇÊ¿ä ¾øÀÌ º¯¼ö¸¸ »ç¿ëÇÒ ¼ö ÀÖ´Ù. from .¸ðµâ import * À» »ç¿ëÇØµµ µÈ´Ù.
- ¸ðµâ°ú ÆÐÅ°Áö(\_\_init__)ÀÇ Ã¹ÁÙ¿¡ '''¸ðµâÀÇ µ¶½ºÆ®¸µ'''Ã³·³ µ¶½ºÆ®¸µÀ» ³ÖÀ» ¼ö ÀÖ°í, ÀÌ°É Ãâ·ÂÇÏ·Á¸é .__doc__¸¦ Ãâ·ÂÇÏ¸é µÈ´Ù.

# collections
***
- collections : Ç¥¿Í ÀÎµ¦½Ìµî µ¥ÀÌÅÍ¸¦ ´Ù·ê¶§ Æ¯È÷ À¯¿ëÇÑ ¶óÀÌºê·¯¸® ¸ðµâ. µ¥ÀÌÅÍÃ³¸®¸¦ À§ÇÑ À¯¿ëÇÑ °´Ã¼°¡ ¸¹ÀÌ ÀÖÀ½. dict()ÀÇ ±âº» API»ç¿ë°¡´É.
- collections.Counter() : Ä«¿îÅÍ°´Ã¼ »ý¼º. µñ¼Å³Ê¸®¿Í µ¿ÀÏÇÏ°Ô Å°/°ª Çü½ÄÀ¸·Î µÇ¾îÀÖÀ¸³ª ÃÖÃÊ Å° Ãß°¡½Ã 0À¸·Î ÀÚµ¿ÃÊ±âÈ­µÊ. 
  .mostcommon(n)À¸·Î ºóµµ¼ö ¼ø Á¤·Ä·Î ¹ÞÀ» ¼ö ÀÖÀ½(n»ý·«°¡´É).
- collections.defaultdict(list) : defaultdict°´Ã¼ »ý¼º. ÇÏ³ªÀÇ Å°¸¦ ¿©·¯ °ª¿¡ ¸ÊÇÎ°¡´É. ÀÎÀÚ·Î ÁÖ¾îÁø °´Ã¼ÀÇ ±âº»°ªÀ» ÃÊ±ê°ªÀ¸·Î »ç¿ë.
- collections.deque(maxlen=n) : deque°´Ã¼ »ý¼º. ¸¶Áö¸· n°³ÀÇ °´Ã¼¸¸À» À¯Áö.   

# appendix
***

## time
***
- time.time() : ÇöÀç½Ã°£ ¹ÝÈ¯. ´ÜÀ§´Â sec. ÇÔ¼ö ½ÃÀÛ Àü°ú ÈÄÀÇ Â÷ÀÌ¸¦ ÀÌ¿ëÇØ ÇÔ¼öÀÇ ½ÇÇà½Ã°£À» ±¸ÇÒ ¼ö µµ ÀÖÀ½.
- time.sleep(n) : sleep. nÃÊ¸¦ sleepÇÔ. nÀº ½Ç¼öµµ °¡´É.
- time ¸ðµâÀÇ localtime ÇÔ¼ö¸¦ ÀÌ¿ëÇÏ¸é UTC °¡ ¾Æ´Ï¶ó KST ·Î, ³¯Â¥¿Í ½Ã°£ ÇüÅÂ·Î º¯È¯ÇØÁØ´Ù.
- time.strftime('Æ÷¸Ë', time.localtime(time.time())) ·Î ¿øÇÏ´Â Æ÷¸ËÀ¸·Î »ç¿ëÇÒ ¼ö ÀÖ´Ù.

## byte
***
- bytes : ¹ÙÀÌÆ® ´ÜÀ§ÀÇ °ªÀ» ¿¬¼ÓÀûÀ¸·Î ÀúÀåÇÏ´Â ½ÃÄö½º °´Ã¼.
- bytes(±æÀÌ): Á¤ÇØÁø ±æÀÌ¸¸Å­ 0À¸·Î Ã¤¿öÁø ¹ÙÀÌÆ® °´Ã¼¸¦ »ý¼º
- bytes(¹Ýº¹°¡´ÉÇÑ°´Ã¼): ¹Ýº¹ °¡´ÉÇÑ °´Ã¼·Î ¹ÙÀÌÆ® °´Ã¼¸¦ »ý¼º ÀÇ µÎ°¡Áö ¹æ¹ýÀ¸·Î ¸¸µé ¼ö ÀÖ´Ù,
- ''³ª ""¾Õ¿¡ b¸¦ ºÙÀÌ¸é ¹ÙÀÌÆ®°´Ã¼°¡ µÊ

### byte array
***
- bytearray : bytes ¿Í °°Áö¸¸ ¿ä¼Òº¯°æ °¡´É.
- bytearray(): ºó ¹ÙÀÌÆ® ¹è¿­ °´Ã¼¸¦ »ý¼º
- bytearray(±æÀÌ): Á¤ÇØÁø ±æÀÌ¸¸Å­ 0À¸·Î Ã¤¿öÁø ¹ÙÀÌÆ® ¹è¿­ °´Ã¼¸¦ »ý¼º
- bytearray(¹Ýº¹°¡´ÉÇÑ°´Ã¼): ¹Ýº¹ °¡´ÉÇÑ °´Ã¼·Î ¹ÙÀÌÆ® ¹è¿­ °´Ã¼¸¦ »ý¼º

## encoding
***
- ÆÄÀÌ½ã¿¡¼­ ¹®ÀÚ¿­ÀÇ ±âº» ÀÎÄÚµùÀº UTF-8ÀÎµ¥, ¹ÙÀÌÆ®°´Ã¼·Î ¸¸µé¸é ASCII ÄÚµå·Î ÀúÀåÇØ ±×°É·Î Ã³¸®ÇÏ°í ½ÍÀ» ¶§ ¹ÙÀÌÆ® °´Ã¼¸¦ »ç¿ëÇÑ´Ù.
- .encode() : ¹®ÀÚ¿­À»  ¹ÙÀÌÆ® °´Ã¼·Î ¹Ù²Ü¶§ »ç¿ë. ("ÀÎÄÚµù")Ã³·³ ÀÎÄÚµùÀ» ÁöÁ¤ÇØÁÖ¸é ÇØ´ç ÀÎÄÚµùÀ¸·Î µÈ ¹ÙÀÌÆ® °´Ã¼·Î ¸¸µê.
- .decode() : ¹ÙÀÌÆ® °´Ã¼¸¦ ¹®ÀÚ¿­·Î ¹Ù²Þ. ÀÌ¶§µµ µÚ¿¡ Æ¯Á¤ ÀÎÄÚµùÀ» ÁöÁ¤ÇÏ¸é, ±× ÀÎÄÚµùÀ¸·Î µÈ ¹ÙÀÌÆ® °´Ã¼¸¦ µðÄÚµùÇÔ.
- bytes("°ª", encoding='ÀÎÄÚµù') ½ÄÀ¸·Î ÀÎÄÚµùÀ» ÁöÁ¤ÇÏ¿© °´Ã¼¸¦ »ý¼ºÇÒ ¼ö ÀÖ´Ù.

## other
***
- eval('¹®ÀÚ¿­') : ¹®ÀÚ¿­ ÇüÅÂÀÇ ÆÄÀÌ½ã ÄÚµå¸¦ ½ÇÇàÇÏ°í °á°ú¸¦ ¹ÝÈ¯ / repr(°´Ã¼) > ÆÄÀÌ½ã ÀÎÅÍÇÁ¸®ÅÍ¿¡¼­ ½ÇÇàÇÒ ¼ö ÀÖ´Â ¹®ÀÚ¿­À» ¹ÝÈ¯
- chr(ÄÚµå°ª) : ASCII ÄÚµå°ª¿¡ ÇØ´çÇÏ´Â ¹®ÀÚ¸¦ ¹ÝÈ¯ / ord(¹®ÀÚ) > ¹®ÀÚÀÇ ASCII ÄÚµå¸¦ ¹ÝÈ¯
- hex(Á¤¼ö) : 16Áø¼ö / oct(Á¤¼ö) > 8Áø¼ö (µÑ´Ù ¹®ÀÚ¿­·Î)
- bin(Á¤¼ö) : 2Áø¼ö º¯È¯ / int('2Áø¼ö¹®ÀÚ¿­', 2) > 2Áø¼ö 10Áø¼ö º¯È¯
- math.isclose(0.1 + 0.2, 0.3) : µÎ ½Ç¼ö°¡ °°ÀºÁö ÆÇ´Ü.
- ÆÄÀÌ½ãÀÇ Á¦ÀÛÀÚ´Â ±Íµµ ¹Ý ·Î¼¶.

# Virtual environment
***
- python -m venv °¡»óÈ¯°æÀÌ¸§ À» °¡»óÈ¯°æÀ» ¸¸µé Æú´õ¿¡¼­ »ç¿ëÇÏ¸é °¡»óÈ¯°æ Æú´õ »ý¼º
- ±× Æú´õ·Î ÀÌµ¿ÇØ .\Scripts\Activate.bat ÆÄÀÏÀ» ½ÇÇàÇÏ¸é(È¤Àº activate ½ÇÇà)°¡»ó È¯°æÀÌ È°¼ºÈ­µÊ. ÀÌ ¸í·ÉÀ» ½ÇÇàÇÑ ÆÄÀÌ½ãÀÇ ¹öÀüÀÌ °¡»óÈ¯°æÀÇ ¹öÀüÀÌ µÊ.
- ¼º°øÀûÀ¸·Î °¡»óÈ¯°æÀ» ¸¸µé¸é (°¡»óÈ¯°æÀÌ¸§)ÀÌ Ç¥½ÃµÇ´Âµ¥, ±× »óÅÂ¿¡¼­ pip ·Î ÆÐÅ°Áö¸¦ ¼³Ä¡ÇÏ¸é \Lib\site-packages ¾È¿¡ ÆÐÅ°Áö°¡ ÀúÀåµÇ¸ç ±× ÆÐÅ°Áö¿¡¼­¸¸ »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- [pip freeze > requirements.txt] : requirements.txt ÆÄÀÏ¿¡ ¼³Ä¡µÈ ÆÐÅ°Áö ¸ñ·Ï ÀúÀå. ±× ¸ñ·Ï´ë·Î ¼³Ä¡ÇÏ·Á¸é pip install -r requirements.txt , »èÁ¦ÇÏ·Á¸é uninstall.
- °¡»óÈ¯°æ Æú´õ¸¦ ¿Å°å´Ù¸é activate.bat, Activate.ps1, activate ÆÄÀÏ ¾ÈÀÇ VIRTUAL_ENV ºÎºÐÀ» ÀÌµ¿½ÃÅ² Æú´õ °æ·Î·Î ¼öÁ¤.
- PyCharm ¿¡¼­ °¡»ó È¯°æÀ» »ç¿ëÇÏ·Á¸é File > Settings... > Project > Project Interpreter ¿¡¼­ ¿À¸¥ÂÊÀÇ Åé´Ï¹ÙÄû ¹öÆ°À» Å¬¸¯ÇÏ°í, 
  Add Local À» Å¬¸¯ÇÏ°í °¡»ó È¯°æÀÇ ÆÄÀÌ½ã ÀÎÅÍÇÁ¸®ÅÍ(python.exe)¸¦ Ãß°¡ÇØÁÖ¸é µÊ.

# json
***
- json : json ¸ðµâ import ÈÄ .json Çü½ÄÀÇ ÆÄÀÏÀ» ¿­¾î json.load(ÆÄÀÏ°´Ã¼)·Î ÆÄÀÌ½ãÀÇ °´Ã¼¿¡ ÀúÀåÇÒ ¼ö ÀÖ´Ù.
- ÆÄÀÌ½ã °´Ã¼¸¦ json ¹®ÀÚ¿­·Î º¯È¯ÇÏ·Á¸é json.dump(°´Ã¼-µñ¼Å³Ê¸®?)·Î Åë°ú½ÃÄÑÁà¾ß ÇÑ´Ù. °¡µ¶¼ºÀ» À§ÇØ 
  (°´Ã¼, indent=µé¿©¾²±âÇÒ ¼ýÀÚ,sort_keys=True(Å° Áß½ÉÀ¸·Î Á¤·Ä))µîÀ» »ç¿ëÇÒ ¼ö ÀÖ´Ù.
- API : API ÀÇ url À» ¹®ÀÚ¿­¿¡ ÀúÀåÇÑ ÈÄ request ¸ðµÑÀÇ get(url) ÇÔ¼ö·Î º¯È¯ÇÑ ÈÄ, ±×°É ´Ù½Ã ÀÐÀ» ¼ö ÀÖ°Ô get ÇÑ °´Ã¼.text ·Î º¯È¯ÇÏ°í, 
  ±×°É json.load ·Î json ÆÄÀÏ·Î º¯È¯ÇÑ´Ù. ±× ÈÄ, API ÀÇ ÂüÁ¶ ÆÄÀÏÀ» Âü°íÇØ µ¥ÀÌÅÍÀÇ Á¾·ù¸¦ ÆÄ¾ÇÇÑ´Ù.

# argparse
- argparse : ¸í·ÉÇà ÀÎÅÍÆäÀÌ½º¸¦ ½±°Ô ÀÛ¼ºÇÏµµ·ÏÇÔ. sys.argv¸¦ ¾î¶»°Ô ÆÄ½ÌÇÒÁö ÆÄ¾ÇÇÏ¸ç, µµ¿ò¸»°ú »ç¿ë¹ý¸Þ¼¼Áö¸¦ ÀÚµ¿ »ý¼ºÇÏ°í, Àß¸øµÈ ÀÎÀÚ¸¦ ÁÙ ¶§ ¿¡·¯¸¦ ¹ß»ý½ÃÅ´.
- parser = argparse.ArgumentParser() : ArgumentParser°´Ã¼ »ý¼º. description(µµ¿ò¸» Àü¿¡ Ç¥½ÃµÉ ÅØ½ºÆ®)µîÀÇ ÀÎÀÚ »ç¿ë°¡´É. 
- parser.add_argument(¿É¼Ç¸í) : ÇÁ·Î±×·¥ ÀÎÀÚ¿¡ ´ëÇÑ Á¤º¸¸¦ Ãß°¡. ¸í·ÉÇàÀÇ ¹®ÀÚ¿­À» °´Ã¼·Î º¯È¯ÇÏ´Â ¹æ¹ýÀ» ¾Ë·ÁÁÜ. ¿É¼Ç¸íÀº ¸®½ºÆ®·Î ¿©·¯°³¸¦ ÁöÁ¤ÇØ ÁÙ ¼öµµ ÀÖÀ½.
- add_argumentÀÎÀÚ : action(ÀÎÀÚ¹ß°ß½Ã ¼öÇàÇÒ ¾×¼ÇÀÇ ±âº»Çü), nargs(¼ÒºñµÇ¾ßÇÏ´Â ÀÎÀÚÀÇ ¼ö), const(ÀÏºÎaction¹×nargs¼±ÅÃ½Ã ÇÊ¿ä»ó¼ý°ª), 
  default(ÀÎÀÚ°¡ ¸í·ÉÇà¿¡µµ namespace¿¡µµ ¾ø´¤À¸¸é »ý¼ºµÇ´Â °ª), type(¸í·ÉÇàÀÎÀÚ°¡ º¯È¯µÇ¾ß ÇÒ Çü), choices(ÀÎÀÚ·Î Çã¿ëµÇ´Â °ªÀÇ ÄÁÅ×ÀÌ³Ê),
  required(¸í·ÉÇà ¿É¼Ç »ý·«°¡´É¿©ºÎ), help(ÀÎÀÚ±â´É¿¡ ´ëÇÑ °£´ÜÇÑ ¼³¸í), metavar(»ç¿ë¸Þ¼¼Áö¿¡ »ç¿ëµÇ´Â ÀÎÀÚ¸í), dest(parse_args()ÀÇ ¹ÝÈ¯°´Ã¼¿¡ Ãß°¡µÉ ¼Ó¼ºÀÌ¸§).
- args = parser.parse_args() : ÀÎÀÚ¸¦ ÆÄ½Ì. ¸í·ÉÇà °Ë»ç -> ÀÎÀÚ¸¦ ÀûÀýÇÑ ÇüÀ¸·Î º¯È¯ -> ÀûÀýÇÑ ¾×¼ÇÀ» È£Ãâ. sys.argv¿¡¼­ ÀÚµ¿À¸·Î ¸í·ÉÇàÀÎÀÚ °áÁ¤.
- args.accumulate(args.ÀÎÀÚ¸í) : ÇØ´ç ÀÎÀÚÀÇ °ªÀ» °¡Á®¿È.

- sys.argv : ÆÄÀÌ½ã ½ºÅ©¸³Æ®¿¡ Àü´ÞµÈ ¸í·ÉÁÙ ÀÎÀÚÀÇ ¸®½ºÆ®. argv[0\]Àº ½ºÅ©¸³Æ® ÀÌ¸§, ÀÎÅÍÇÁ¸®ÅÍ¿¡ ÀÌ¸§ÀÌ Àü´ÞµÇÁö ¾ÊÀ¸¸é ºó ¹®ÀÚ¿­.

# threading | º´·Ä½ÇÇà
- threading : ±âº»ÀûÀ¸·Î ÇÏ³ªÀÇ ¸ÞÀÎ¾²·¹µå°¡ ÄÚµå¸¦ ¼øÂ÷ÀûÀ¸·Î ½ÇÇàÇÏ´Â ÆÄÀÌ½ã¿¡¼­ ÄÚµå¸¦ º´·Ä·Î ½ÇÇàÇÏ±â À§ÇØ º°µµÀÇ ¾²·¹µå¸¦ »ý¼ºÇÏ´Â ¸ðµâ.
  ÆÄÀÌ½ã¿¡´Â Àü¿ª ÀÎÅÍÇÁ¸®ÅÍ ¶ôÅ·ÀÌ ÀÖ¾î, Æ¯Á¤ ½ÃÁ¡¿¡ ÇÏ³ªÀÇ ÄÚµå¸¸ ½ÇÇàÇÏ´Âµ¥, ÀÌ ¶§¹®¿¡ ÀÎÅÍ¸®ºù ¹æ½ÄÀ¸·Î ÄÚµå¸¦ ºÐÇÒÇØ ½ÇÇàÇÏ¸ç,
  ´ÙÁß CPU¿¡¼­ÀÇ º´·Ä½ÇÇàÀ» À§ÇØ¼­´Â multiprocessing(´ÙÁßÇÁ·Î¼¼½º ÀÌ¿ë)¸ðµâÀ» »ç¿ëÇØ¾ß ÇÔ.
- t = threading.Thread() : ¾²·¹µå °´Ã¼ È£Ãâ. targetÀÎÀÚ¿¡ ¾²·¹µå°¡ ½ÇÇàÇÒ ÇÔ¼ö¸¦ ÁÖ°í, args(kwargs)ÀÎÀÚ¿¡ ÇØ´ç ÇÔ¼öÀÇ ÀÎÀÚ¸¦ ³Ö¾î ¾²·¹µå¸¦ ½ÇÇà ÇÒ ¼ö ÀÖÀ½.
- ÆÄ»ýÅ¬·¡½º Á¦ÀÛ : threading.Thread¸¦ »ó¼ÓÇÏ´Â Å¬·¡½º¸¦ Á¦ÀÛ ÈÄ run(¾²·¹µå°¡ ½ÇÁ¦ ½ÇÇàÇÏ´Â ¸Þ¼­µå)¸¦ ÀçÁ¤ÀÇÇÏ´Â ¹æ½ÄÀ¸·Îµµ »ç¿ëÀÌ °¡´ÉÇÔ.
- t.start() : ¾²·¹µå¸¦ ½ÇÇàÇÔ.

- µ¥¸ó¾²·¹µå : ¹é±×¶ó¿îµå¿¡¼­ ½ÇÇàµÇ´Â, ¸ÞÀÎ¾²·¹µå°¡ Á¾·áµÇ¸é Áï½Ã Á¾·áµÇ´Â ¾²·¹µå. ÀÏ¹Ý ¾²·¹µå´Â ¸ÞÀÎÀÌ ³¡³ªµµ ÀÚ½ÅÀÇ ÀÛ¾÷ÀÌ ³¡³¯¶§±îÁö °è¼Ó ½ÇÇà.
- t.deamon = True : µ¥¸ó¾²·¹µå·Î ¼³Á¤.

# git
***
- git clone (±êÇãºê µð·ºÅä¸® ÁÖ¼Ò) .(ÇöÀç µð·ºÅä¸®) : github µð·ºÅä¸®ÀÇ ÆÄÀÏÀ» ÀüºÎ º¹»çÇØ¿È
- git diff : ¸¶Áö¸· ¼öÁ¤ ÀÌÈÄ Ãß°¡ÇÑ ³»¿ëÀ» ¾Ë·ÁÁÜ
- git add (´õÇÒ ÆÄÀÏ) : ÇÊµå(Ä¿¹Ô½Ã git¿¡ µî·ÏµÉ ÆÄÀÏÀÌ ÀÖ´Â °ø°£)¿¡ ¿Ã¸²
- git commit -m "(¹öÀü-¼³¸í-)" : ÄÄÇ»ÅÍ ³»ÀÇ git¿¡ µî·Ï.
- git log : »ý¼ºÇÑ ¹öÀü º¸¿©ÁÜ
- git push : ÄÄÇ»ÅÍ¿¡ ÀúÀåµÈ gitÆÄÀÏÀ» ±êÇãºê¿¡ µî·Ï.
- git status : »óÅÂ(¼öÁ¤ÇÑ ºÎºÐ)º¸¿©ÁÜ

- .gitignore : ±ê¿¡¼­ °ü¸®ÇÏÁö ¾ÊÀ», ¹«½ÃÇÒ ÆÄÀÏ/µð·ºÅä¸®¸¦ Á¤ÀÇÇÑ ÆÄÀÏ. ÇÁ·ÎÁ§Æ® ÃÖ»óÀ§¿¡ Á¸ÀçÇØ¾ß ÇÔ.
- ±ÔÄ¢ : '#'·Î ½ÃÀÛÇÏ´Â ¶óÀÎÀº ¹«½Ã | Æò¹üÇÑ ÆÄÀÏÇü½Ä(*.aµî)»ç¿ë | '/'·Î ½ÃÀÛÇÏ¸é ÇÏÀ§ µð·ºÅä¸®¿¡ ¹ÌÀû¿ë('/dir/\*.a'¸é /dir/subdir/ÀÇ aÆÄÀÏÀº ¹ÌÀû¿ë)
  | µð·ºÅä¸®´Â '/'¸¦ ³¡¿¡ »ç¿ëÇØ Ç¥½Ã(¾øÀ¸¸é È®ÀåÀÚ°¡ ¾ø´Â ÆÄÀÏÃ³¸®) | '!'·Î ½ÃÀÛÇÏ´Â ÆÐÅÏÀÇ ÆÄÀÏÀº ¹«½ÃÇÏÁö ¾ÊÀ½ | ÆÄÀÏ/µð·ºÅä¸®µîÀ» ÀÔ·Â½Ã ÇØ´ç ÆÄÀÏ/µð·ºÅä¸®¸¦ ¹«½Ã.

- git lfs(LargeFileStorage) : git¿¡¼­ 100MBÀÌ»óÀÇ ÆÄÀÏÀ» °ü¸®ÇÏ±âÀ§ÇÑ ÇÁ·Î±×·¥. add/pushÀÌÀü¿¡ »ç¿ëÇØÁà¾ß ÇÔ.
- git lfs install : ½ÇÇàÇÑ ¸®Æ÷ÁöÅä¸®¸¦ lfsÀÇ °ü¸®ÇÏ¿¡ ³ÖÀ½.
- git lfs track "*.È®ÀåÀÚ" : ÇØ´ç È®ÀåÀÚÀÇ ÆÄÀÏÀ» °ü¸®ÇÔ.

- BFG Repo-Cleaner : ¿øÄ¡ ¾Ê´Â µ¥ÀÌÅÍ Á¦°Å µîÀ» Áö¿øÇÏ´Â ¿ÀÇÂ¼Ò½º µµ±¸.
- [java -jar (bfgÆÄÀÏ°æ·Î+bfgÆÄÀÏ¸í).jar --strip-blobs-bigger-than 100M (.gitÆÄÀÏ¸í).git] : ÇØ´ç ±ê ·¹Æ÷ÁöÅä¸® »ç¿ë. Á¦°ÅÇÒ ÆÄÀÏÀÌ ÀÖ´Â ·¹Æ÷ÁöÅä¸®¸¦ »ç¿ëÇØ¾ß ÇÔ.
- [java -jar (bfgÆÄÀÏ°æ·Î+bfgÆÄÀÏ¸í).jar --delete-files(-D) (»èÁ¦ÇÒÆÄÀÏ)] : »èÁ¦ ÆÄÀÏ µî·Ï. ÆÄÀÏ »èÁ¦.
- [git reflog expire --expire=now --all && git gc --prune=now --aggressive] : ÆÄÀÏ »èÁ¦ È®Á¤. ÀÌ ÀÌÈÄ push¸¦ ½ÇÇàÇÏ¸é µÊ. 
