#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;




public class CustomImportSettings : AssetPostprocessor
{	
	

    void OnPreprocessModel()
    {
        ModelImporter modelImporter = assetImporter as ModelImporter;
        modelImporter.generateSecondaryUV = true;
        modelImporter.addCollider = true; 
        modelImporter.importAnimation = false;

        
    }
    
}

#endif