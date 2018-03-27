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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Vector;

/**
 * Created by pf on 3/25/18.
 */

public class TF_Caculation_Thread extends Thread {

    static final String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";
    static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    static final String INPUT_NODE = "input";
    static final String OUTPUT_NODE = "output";
    static final int inputSize = 224;
    static final int imageMean = 117;
    static final int imageStd = 1;
    static final int numClasses = 2000;

    private Bitmap bitmap;
    private String[] OUTPUT_NODES = new String[] {OUTPUT_NODE};
    private TensorFlowInferenceInterface inferenceInterface=null;
    private Handler threadHandler;
    private Vector<String> labels = new Vector<>();

    TF_Caculation_Thread(Bitmap btm, AssetManager assetManager, Handler mHandler){
        super();
        bitmap = btm;
        threadHandler = mHandler;
        BufferedReader br =null;
        String actualFilename = LABEL_FILE.split("file:///android_asset/")[1];
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line;
            while((line = br.readLine()) != null){
                labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
    }

    public float[] classify(Bitmap btm){
        //int array_length = input_data.length;
        float[] output_array = new float[numClasses];
        int width = btm.getWidth();
        int height = btm.getHeight();

        //prepare the data: convert the format of bitmap into 2 dimension array
        int[] intValues = new int[width * height] ;//= new int[inputSize*inputSize];
        float[] floatValues = new float[inputSize*inputSize*3];

        btm.getPixels(intValues, 0, btm.getWidth(), 0, 0, btm.getWidth(), btm.getHeight());
        Log.e("test", String.format("W H: %d %d",width,height));

        //如果照片的像素 数目大于inputSize^2,则舍去多余部分,相反,则用0补足
        for (int i = 0; i < inputSize*inputSize; ++i) {
            final int val;
            if(i<width*height)       //未越界
                val = intValues[i];
            else                   //越界用0填充
                val =0;
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        inferenceInterface.feed(INPUT_NODE, floatValues, 1, inputSize, inputSize, 3);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE,output_array);
        return  output_array;
    }

    @Override
    public void run(){
        float[] result_array ;
        float max_value=0;
        int max_index=0;

        try{
            result_array = classify(bitmap);
            for (int i=0; i< result_array.length; i++) {
                if(result_array[i] > max_value){
                    max_value= result_array[i];
                    max_index = i;
                }
            }
            Log.e("test",String.format("max_value: %f \n max_indec: %d",max_value, max_index));
            //Log.e("test",String.format("result is %f",result_array[0]));

            Message msg = new Message();
            msg.what =0;
            msg.obj = labels.get(max_index);
            threadHandler.sendMessage(msg);

        }catch(Exception e){
            e.printStackTrace();
            Log.e("test","err");
        }
    }
}
