package com.example.stock

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main3.*



class Main3Activity : AppCompatActivity() {

    @SuppressLint("ResourceAsColor")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main3)

            val view = this.getWindow().getDecorView()
            view.setBackgroundColor(android.R.color.black )


        button4.setOnClickListener(){
            val intent = Intent(this, Main4Activity::class.java)
            startActivity(intent)
        }
    }
}
