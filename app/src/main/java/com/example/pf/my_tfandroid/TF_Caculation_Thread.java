package com.example.pf.my_tfandroid;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Created by pf on 3/25/18.
 */

public class TF_Caculation_Thread extends Thread {

    static final String MODEL_FILE = "file:///android_asset/classify_image_graph_def.pb";
    static final String INPUT_NODE = "DecodeJpeg/contents";
    static final String OUTPUT_NODE = "softmax";
    static final int inputSize = 224;
    static final int imageMean = 117;
    static final int imageStd = 1;
    static final int numClasses = 2000;



    public Bitmap bitmap;
    public TextView tv;
    public ImageView iv;
    public String[] OUTPUT_NODES = new String[] {OUTPUT_NODE};
    TensorFlowInferenceInterface inferenceInterface=null;

    TF_Caculation_Thread(Bitmap btm, AssetManager assetManager, TextView tv_argue, ImageView iv_argue){
        super();
        bitmap = btm;
        tv = tv_argue;
        iv = iv_argue;
        inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
    }

    public float[] classify(Bitmap btm){
        //int array_length = input_data.length;
        int array_length = 1;
        float[] output_array = new float[numClasses];

        //prepare the data: convert the format of bitmap into 2 dimension array
        int[] intValues = new int[inputSize*inputSize];
        float[] floatValues= new float[inputSize*inputSize*3];

        Log.e("test ","1");
        //Bitmap.getPixel的定义生成异常
        //btm.getPixels(intValues, 0, btm.getWidth(), 0, 0, btm.getWidth(), btm.getHeight());
        //Log.e("test ","2");
        //for (int i = 0; i < intValues.length; ++i) {
        //    final int val = intValues[i];
        //    floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
        //    floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
        //    floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        //}
        Log.e("test ","3");
        inferenceInterface.feed(INPUT_NODE, floatValues, 1, inputSize, inputSize, 3);
        Log.e("test ","4");
        inferenceInterface.run(OUTPUT_NODES);
        Log.e("test ","5");
        inferenceInterface.fetch(OUTPUT_NODE,output_array);
        Log.e("test ","6");
        return  output_array;
    }

    @Override
    public void run(){
        //result_array =  classify(bitmap);
        try{
            //tv.setText("laji java");
            classify(bitmap);
            Log.e("exception!","in try");
        }catch(Exception e){
            e.printStackTrace();
            Log.e("exception!","err");
        }
    }
}
