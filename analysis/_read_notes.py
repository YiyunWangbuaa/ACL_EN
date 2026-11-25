# -*- coding: utf-8 -*-
import zipfile, re, io
DOC = r"C:\Users\minec\Desktop\ACL方法章节_初版.docx"
z = zipfile.ZipFile(DOC)

# comments
if 'word/comments.xml' in z.namelist():
    from xml.etree import ElementTree as ET
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    root = ET.fromstring(z.read('word/comments.xml'))
    print("=== comments:", len(root.findall(W+'comment')), "===")
    for c in root.findall(W+'comment'):
        cid = c.get(W+'id'); author = c.get(W+'author')
        text = ''.join(t.text or '' for t in c.iter(W+'t'))
        print(f"[{cid}] ({author}): {text}")
else:
    print("no comments.xml")

# document text (paragraph by paragraph)
doc = z.read('word/document.xml').decode('utf-8')
out = []
paras = re.split(r'(?<=</w:p>)', doc)
for pi, p in enumerate(paras):
    t = ''.join(re.findall(r'<w:t[^>]*>(.*?)</w:t>', p))
    # comment refs
    refs = re.findall(r'w:id="(\d+)"', p)
    if t.strip():
        out.append(f"[{pi}] {t}")
io.open(r"C:\Users\minec\Desktop\ACL\workflow\analysis\_text.txt", 'w', encoding='utf-8').write('\n'.join(out))
print("paragraphs:", len(out))
