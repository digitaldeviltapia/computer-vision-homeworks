import numpy as np
import open3d as o3d


    
#modificamos el ransac que ya teníamos para que función con 3 puntos, para crear un plano en puntos 3D
def ransac_prob3(puntos1, num_iter, tolerancia):
    
    T = len(puntos1) + 1 #esto nos desactiva el if del paso iii) de manera definitiva (podemos cambairlo después)
    
    mejor_plano = None #aquí va el array con los coeficientes del mejor plano, lo inicializamos en None
    mejor_conjunto_inliers = [] #para guardar el mejor conjunto de inliers
    
    for i in range(num_iter): #(este es el paso iv))
        # i) randomly select a sample of 3 data points from S and
        #vamos a ordenar el conjunto de puntos de menor a mayor en z, para poder tomar un subconjunto de ese conjuto
        #y así evitar que los puntos estén muy separados, eligiendo una muestra dentro del subconjunto
        
        puntos_ordenados = puntos1[puntos1[:, 2].argsort()]
        num_puntos = len(puntos_ordenados)
        subconjunto = puntos_ordenados[: int(num_puntos * 0.1)]
        indices = np.random.choice(len(subconjunto), 3, replace=False)

        sample1 = subconjunto[indices]
        print("Los puntos seleccionados son:", sample1)
        
    
        #ahora vamos a calcular los coeficientes de la ecuación del plano ax+by+cz+d = 0
        
        #para esto vamos a calcular los vectores normales
        v1 = sample1[1] - sample1[0]
        v2 = sample1[2] - sample1[0]
        
        #ahora calculamos el vector normal
        normal = np.cross(v1, v2)
        
        #ahora calculamos el d
        d = -normal @ sample1[0]
        
        #ahora vamos a normalizar el vector normal
        normal = normal/np.linalg.norm(normal)
        a, b, c = normal
        
        
    
        #ii) determine the set of data points S_i which are within a distance threshold t of the model.
        #the set S_i is the consensus set of the sample and defines the inliers of S.
        
        inliers = []
        for j in range(len(puntos1)):
            #vamos a calcular la distancia de un punto a al plano encntrado
            #print("Estamos en la iteracion", i, "y en el punto", j, "con numero de inliers", len(inliers))
            punto = puntos1[j]
            distancia = np.abs(a*punto[0] + b*punto[1] + c*punto[2] + d)/np.sqrt(a**2 + b**2 + c**2)
            if distancia < tolerancia:
                inliers.append(j)
                
        #iii) If the size of S_i is greater than T (tolerancia), re-estiimate the model ussing all the points in S_i and terminate
        inliers = np.array(inliers)
        numero_inliers = len(inliers)
        print("El numero de inliers es:", numero_inliers)
        
        if numero_inliers > T:
            #volvemos a calcular la homografía y terminamos
            puntos_inliers = puntos1[inliers]
            centroide = np.mean(puntos_inliers, axis=0)
            #utilizaremos la matriz de convarianza para ver la dsitribución de los datos
            cov = np.cov(puntos_inliers.T)
            S_cov, V_cov, D_cov = np.linalg.svd(cov)
            nueva_normal = D_cov[-1]
            nueva_d = -nueva_normal @ centroide
            nuevo_a, nuevo_b, nuevo_c = nueva_normal
            mejor_plano = np.array([nuevo_a, nuevo_b, nuevo_c, nueva_d])
            return mejor_plano, inliers
        
        #iv is the size of S_i is less than T, select a new subtset and repeat above (esto fue el for de todo el codigo)
        #v) After N trials the largest consensus set S_i is selected and he model is re-estimated using all the
        #points in the subset S_i.
        if numero_inliers > len(mejor_conjunto_inliers):
            
            mejor_conjunto_inliers = inliers
            mejor_plano = np.array([a, b, c, d])
            #print(f"Normal del plano: {mejor_plano[:3]}")
            
    if len(mejor_conjunto_inliers) == 0:
        return None, []
            
    #recalculamos el mejor plano usando los puntos en el conjunto de inliers
    puntos_inliers = puntos1[mejor_conjunto_inliers]
    centroide = np.mean(puntos_inliers, axis=0)
    
    cov = np.cov(puntos_inliers.T)
    #print("covarianza calculada con puntos_inliers", cov)
    S_cov, V_cov, D_cov = np.linalg.svd(cov)
    nueva_normal = D_cov[2]
    #print("Vectores propios de la matriz de covarianza", D_cov)
    #print("nueva normal", nueva_normal)
    
    nueva_d = -nueva_normal @ centroide
    nuevo_a, nuevo_b, nuevo_c = nueva_normal
    
    

    mejor_plano = np.array([nuevo_a, nuevo_b, nuevo_c, nueva_d])
    
    #print(f"Normal del plano: {mejor_plano[:3]}")
    #print("centroide", centroide)
    #print("varianza de inliers en x", np.var(puntos_inliers[:, 0]))
    

    return mejor_plano, mejor_conjunto_inliers

##dibujar plano

def dibujar_plano(mejor_plano, punto_central, tamano=50.0, color=[0, 1, 0]):
    a, b, c, d = mejor_plano

    # Generamos dos vectores ortogonales al normal para elegir uno de referencia
    normal = np.array([a, b, c])
    if abs(a) > abs(c):  
        v1 = np.array([-b, a, 0])
    else:
        v1 = np.array([0, -c, b])

    v1 = v1 / np.linalg.norm(v1) * tamano
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2) * tamano

    # vertices de plano
    puntos_plano = np.array([
        punto_central + v1 + v2,
        punto_central + v1 - v2,
        punto_central - v1 - v2,
        punto_central - v1 + v2
    ])

    # triangulos dela malla
    caras = np.array([
        [0, 1, 2],
        [2, 3, 0]
    ])

    # Crear la malla
    plano_mesh = o3d.geometry.TriangleMesh()
    plano_mesh.vertices = o3d.utility.Vector3dVector(puntos_plano)
    plano_mesh.triangles = o3d.utility.Vector3iVector(caras)
    plano_mesh.paint_uniform_color(color)  # Color verde por defecto

    return plano_mesh


#vamos a correr todo
# leemos 
pcd = o3d.io.read_point_cloud("toronto.ply")

# convertir a numpy array
point_cloud = np.asarray(pcd.points)
print("el numero de puntos es:", len(point_cloud))  # Número de puntos (N)

#vamos a calcular el centroide de la nube
centroide = np.mean(point_cloud, axis=0)
print("El centroide de la nube es:", centroide)
point_cloud_centrada = point_cloud - centroide
# y rescalamos 
escala = np.max(np.abs(point_cloud_centrada), axis=0)
point_cloud_centrada = point_cloud_centrada/escala

# Parámetros
num_iter = 10  # Iteraciones
tolerancia = 0.01   # Tolerancia

# mandamos a llamar nuestra función ransac
mejor_plano, inliers = ransac_prob3(point_cloud_centrada, num_iter, tolerancia)
#y sacamos el plano 
plano_mesh = dibujar_plano(mejor_plano, centroide, tamano=300.0, color=[0, 1, 0])
if mejor_plano is None:
    print("No se encontró un plano en la nube de puntos.")
else:
    print(f"Plano detectado: {mejor_plano}")

    # nube de inliers
    inlier_cloud = point_cloud[inliers]
    
    # visualización de los inliers
    inlier_pcd = o3d.geometry.PointCloud()
    inlier_pcd.points = o3d.utility.Vector3dVector(inlier_cloud)
    
    # marcaremos los inliners en el dibujo
    inlier_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Rojo
    
    # visualización
    o3d.visualization.draw_geometries([pcd, inlier_pcd, plano_mesh])
