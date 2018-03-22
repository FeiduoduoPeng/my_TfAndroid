package com.example.pf.my_tfandroid;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Picture;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("tensorflow_inference");
    }
    static final int REQUEST_CODE_IMAGE_CAPTURE = 1;
    static final String MODEL_FILE = "file:///android_asset/easy_pb.pb";
    static final String INPUT_NODE = "input_node:0";
    static final String OUTPUT_NODE = "output_node:0";
    private ImageView imageView;
    private TextView textView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        if(requestCode == REQUEST_CODE_IMAGE_CAPTURE && resultCode == RESULT_OK){
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            //imageView.setImageBitmap(imageBitmap);

            new TF_Caculation_Thread(imageBitmap).start();
        }
    }

    private void dispatchTakePictureIntent(){
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if(takePictureIntent.resolveActivity(getPackageManager()) != null){
            startActivityForResult(takePictureIntent, REQUEST_CODE_IMAGE_CAPTURE);
        }
    }

    public void takePhoto(View v){
        dispatchTakePictureIntent();

    }

    class TF_Caculation_Thread extends Thread{

        public String[] output_nodes_array = {OUTPUT_NODE};
        TensorFlowInferenceInterface inferenceInterface=null;

        TF_Caculation_Thread(Bitmap bitmap){
            super();
            imageView.setImageBitmap(bitmap);
            inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
        }

        public float[] classify(float[] input_data){
            int array_length = input_data.length;
            float[] output_array = new float[array_length];
            inferenceInterface.feed(INPUT_NODE,input_data ,array_length, 1);
            inferenceInterface.run(output_nodes_array);
            inferenceInterface.fetch(OUTPUT_NODE,output_array);
            return  output_array;
        }
        @Override
        public void run(){
            float[] input_array ={3,4};
            float[] result_array;
            try{
                result_array = classify(input_array);
                textView.setText(String.format("Result:\n %f \n %f",result_array[0], result_array[1]));
                //textView.setText(String.format("Result:\n %f",10.0));
            }catch(Exception e){
                e.printStackTrace();
            }
        }
    }
}
