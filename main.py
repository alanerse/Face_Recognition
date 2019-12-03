import cv2
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

#Link planilha : https://docs.google.com/spreadsheets/d/1X6AU5cHoqIRq-9M_13W3ELoQdKPDhBl_Y1eY1Xv_DY8/edit?usp=sharing

#Variáveis-------------------------------------------------------------
scope = ['https://spreadsheets.google.com/feeds']

credentials = ServiceAccountCredentials.from_json_keyfile_name('tpdi-260620-b0d4d2abab65.json', scope)

gc = gspread.authorize(credentials)

wks = gc.open_by_key('1X6AU5cHoqIRq-9M_13W3ELoQdKPDhBl_Y1eY1Xv_DY8')

worksheet = wks.get_worksheet(0)

conversion = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

nomes = worksheet.col_values(1)

nomes.remove('Nomes')

presentes = {
}
for nome in nomes:
    presentes[nome] = False

#Funções---------------------------------------------------
def planilha(presentes):

    # Atualiza celula
    aux = 2
    aulas = worksheet.row_values(2)
    aula = conversion[len(aulas)]
    for i in presentes:
        if presentes[i] == True:
           worksheet.update_acell(aula + str(aux), 'X')
        else:
            worksheet.update_acell(aula + str(aux), 'F')
        aux+=1

#-------------------------------------------------------------

hog = cv2.HOGDescriptor()
face_cascade = cv2.CascadeClassifier('haar.xml')

img1 = cv2.imread('figs/alanjonata.jpeg')

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:

    rosto = img1[y:y + h, x:x + w]

    if w > 100 and h > 100:
        rosto = cv2.resize(rosto, (160, 160))
        rosto = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)

    h1 = hog.compute(rosto)
    soma = 0

    dic = {'nome':'a', 'd':100000000000000.0}

    for dirPrincipal, nomeDirs, nomeArqs in os.walk('images'):
        for subDir in nomeDirs:
            caminhoPasta = os.path.join(dirPrincipal, subDir)
            for filename in os.listdir(caminhoPasta):
                caminhoAbs = caminhoPasta + "/" + filename
                img2 = cv2.imread(caminhoAbs)
                img2 = cv2.resize(img2, (160, 160))
                h2 = hog.compute(img2)

                soma = 0
                for i in range(len(h1)):
                    soma += ((h1[i] - h2[i])**2)

                dist = soma ** (1/2)
                valor = dist[0].item()
                print(str(valor))

                if dic['d'] > valor:
                    dic['d'] = valor
                    dic['nome'] = filename

    dic['nome'] = dic['nome'].replace('.jpg',"")
    dic['nome'] = dic['nome'].replace('.jpeg',"")
    dic['nome'] = dic['nome'].replace('.PNG',"")

    print(dic['nome'] + str(dic['d']))
    presentes[dic['nome']] = True

planilha(presentes)
