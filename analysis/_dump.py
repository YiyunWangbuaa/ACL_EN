# -*- coding: utf-8 -*-
import zipfile, re, io
DOC = r"C:\Users\minec\Desktop\ACL方法章节_初版.docx"
z = zipfile.ZipFile(DOC)
doc = z.read('word/document.xml').decode('utf-8')
out = []
for p in re.split(r'(?<=</w:p>)', doc):
    t = ''.join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', p))
    if t.strip():
        out.append(t)
io.open(r"C:\Users\minec\Desktop\ACL\workflow\analysis\_text.txt", 'w', encoding='utf-8').write('\n'.join(out))
print("ok", len(out))
