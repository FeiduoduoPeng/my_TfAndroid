package com.example.pf.my_tfandroid;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
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
    private Bitmap resized_bitmap;
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

    private static Bitmap createSquaredBitmap(Bitmap srcBmp, Bitmap.Config cfg) {
        int dim = Math.max(srcBmp.getWidth(), srcBmp.getHeight());
        Bitmap dstBmp = Bitmap.createBitmap(dim, dim, cfg);

        Canvas canvas = new Canvas(dstBmp);
        canvas.drawColor(Color.WHITE);
        canvas.drawBitmap(srcBmp, (dim - srcBmp.getWidth()) / 2, (dim - srcBmp.getHeight()) / 2, null);

        return dstBmp;
    }

    public float[] classify(Bitmap btm){
        //int array_length = input_data.length;
        float[] output_array = new float[numClasses];
        Log.e("test","before resize");
        Bitmap.Config btm_config = btm.getConfig();
        Bitmap square_bitmap = createSquaredBitmap(btm, btm_config);
        Log.e("test","after resize");

        int width = square_bitmap.getWidth();
        int height = square_bitmap.getHeight();
        Log.e("test", String.format("W H: %d %d",width,height));

        resized_bitmap = Bitmap.createBitmap(square_bitmap,0,0,inputSize,inputSize);

        //prepare the data: convert the format of bitmap into 2 dimension array
        int[] intValues = new int[inputSize * inputSize] ;//= new int[inputSize*inputSize];
        float[] floatValues = new float[inputSize*inputSize*3];

        resized_bitmap.getPixels(intValues, 0, resized_bitmap.getWidth(), 0, 0, resized_bitmap.getWidth(), resized_bitmap.getHeight());

        //如果照片的像素 数目大于inputSize^2,则舍去多余部分,相反,则用0补足
        for (int i = 0; i < inputSize*inputSize; ++i) {
            final int val;
            val = intValues[i];
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

            Message msg_text = new Message();
            msg_text.what = 0;
            msg_text.obj = labels.get(max_index);
            threadHandler.sendMessage(msg_text);

            Message msg_image = new Message();
            msg_image.what = 1;
            msg_image.obj = resized_bitmap;
            threadHandler.sendMessage(msg_image);

        }catch(Exception e){
            e.printStackTrace();
            Log.e("test","err");
        }
    }
}
